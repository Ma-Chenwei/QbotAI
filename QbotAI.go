package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha1"
	"crypto/tls"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	AppName    = "QbotAI"
	AppVersion = "2.0.0"

	ConfigFile = "config.json"
	MemoryFile = "memory.md"

	QQAPIBase   = "https://api.sgroup.qq.com"
	QQTokenBase = "https://bots.qq.com"

	MaxMemoryBytes int64 = 2 * 1024 * 1024 * 1024

	GroupMessageIntent uint64 = 1 << 25

	DefaultWebHost = "127.0.0.1"
	DefaultWebPort = 8080
)

var (
	config   Config
	configMu sync.RWMutex

	memoryMu sync.Mutex

	qqBot *QQBot
	aiBot *AIClient

	startTime = time.Now()

	aiOperationCounter int64

	replyMu sync.Mutex

	replyHistory = make(map[string][]time.Time)

	groupMu sync.Mutex

	groupContexts = make(map[string]*GroupContext)
)

// ============================================================
// CONFIG
// ============================================================

type Config struct {
	QQ struct {
		AppID        string `json:"app_id"`
		ClientSecret string `json:"client_secret"`
		Enabled      bool   `json:"enabled"`
	} `json:"qq"`

	AI struct {
		Enabled      bool   `json:"enabled"`
		BaseURL      string  `json:"base_url"`
		APIKey       string  `json:"api_key"`
		Model        string  `json:"model"`
		SystemPrompt string  `json:"system_prompt"`

		Temperature float64 `json:"temperature"`
		MaxTokens   int     `json:"max_tokens"`

		MemoryEnabled bool `json:"memory_enabled"`
	} `json:"ai"`

	GroupAI struct {
		Enabled bool `json:"enabled"`

		ContextMessages int `json:"context_messages"`

		DecisionDelaySeconds int `json:"decision_delay_seconds"`

		ReplyCooldownSeconds int `json:"reply_cooldown_seconds"`

		MaxRepliesPerHour int `json:"max_replies_per_hour"`

		ProactiveLevel int `json:"proactive_level"`

		OnlyReplyWhenUseful bool `json:"only_reply_when_useful"`
	} `json:"group_ai"`

	Memory struct {
		Enabled bool `json:"enabled"`

		CompressionEvery int `json:"compression_every"`

		CompressionRatio float64 `json:"compression_ratio"`

		MaxBytes int64 `json:"max_bytes"`

		DeleteOldestFirst bool `json:"delete_oldest_first"`
	} `json:"memory"`

	Web struct {
		Host string `json:"host"`
		Port int    `json:"port"`
	} `json:"web"`

	Bot struct {
		Name   string `json:"name"`
		Prefix string `json:"prefix"`
	} `json:"bot"`
}

type Int64Flexible int64

func (v *Int64Flexible) UnmarshalJSON(data []byte) error {
    var n int64

    if err := json.Unmarshal(data, &n); err == nil {
        *v = Int64Flexible(n)
        return nil
    }

    var s string

    if err := json.Unmarshal(data, &s); err != nil {
        return err
    }

    parsed, err := strconv.ParseInt(
        strings.TrimSpace(s),
        10,
        64,
    )

    if err != nil {
        return err
    }

    *v = Int64Flexible(parsed)

    return nil
}

func normalizeGatewayAddress(addr string) string {
	addr = strings.TrimSpace(addr)

	if addr == "" {
		return "api.sgroup.qq.com:443"
	}

	if _, _, err := net.SplitHostPort(addr); err == nil {
		return addr
	}

	return net.JoinHostPort(addr, "443")
}

func defaultConfig() Config {
	var c Config

	c.QQ.AppID = ""
	c.QQ.ClientSecret = ""
	c.QQ.Enabled = false

	c.AI.Enabled = false
	c.AI.BaseURL = "https://api.openai.com/v1"
	c.AI.APIKey = ""
	c.AI.Model = "gpt-4o-mini"
	c.AI.SystemPrompt =
		"你是一个QQ群里的普通成员型AI助手。你应该像一个真实群友一样自然参与聊天。不要为了刷存在感而频繁发言。只有在有必要、能够帮助别人、适合接话、或者你真的有有价值的信息时才发言。如果没有必要发言，必须保持沉默。被@时必须认真回答。"

	c.AI.Temperature = 0.7
	c.AI.MaxTokens = 1200
	c.AI.MemoryEnabled = true

	c.GroupAI.Enabled = true
	c.GroupAI.ContextMessages = 20
	c.GroupAI.DecisionDelaySeconds = 3
	c.GroupAI.ReplyCooldownSeconds = 30
	c.GroupAI.MaxRepliesPerHour = 30
	c.GroupAI.ProactiveLevel = 40
	c.GroupAI.OnlyReplyWhenUseful = true

	c.Memory.Enabled = true
	c.Memory.CompressionEvery = 15
	c.Memory.CompressionRatio = 0.75
	c.Memory.MaxBytes = MaxMemoryBytes
	c.Memory.DeleteOldestFirst = true

	c.Web.Host = DefaultWebHost
	c.Web.Port = DefaultWebPort

	c.Bot.Name = AppName
	c.Bot.Prefix = "/"

	return c
}

func loadConfig() error {
	configMu.Lock()
	defer configMu.Unlock()

	data, err := os.ReadFile(ConfigFile)

	if err != nil {
		if os.IsNotExist(err) {
			config = defaultConfig()
			return saveConfigLocked()
		}

		return err
	}

	if len(bytes.TrimSpace(data)) == 0 {
		config = defaultConfig()
		return saveConfigLocked()
	}

	config = defaultConfig()

	if err := json.Unmarshal(data, &config); err != nil {
		return err
	}

	if config.Memory.MaxBytes <= 0 {
		config.Memory.MaxBytes = MaxMemoryBytes
	}

	if config.Memory.CompressionEvery <= 0 {
		config.Memory.CompressionEvery = 15
	}

	if config.Memory.CompressionRatio <= 0 ||
		config.Memory.CompressionRatio >= 1 {
		config.Memory.CompressionRatio = 0.75
	}

	return nil
}

func saveConfig() error {
	configMu.Lock()
	defer configMu.Unlock()

	return saveConfigLocked()
}

func saveConfigLocked() error {
	data, err := json.MarshalIndent(
		config,
		"",
		"  ",
	)

	if err != nil {
		return err
	}

	tmp := ConfigFile + ".tmp"

	if err := os.WriteFile(
		tmp,
		data,
		0600,
	); err != nil {
		return err
	}

	return os.Rename(
		tmp,
		ConfigFile,
	)
}

func getConfig() Config {
	configMu.RLock()
	defer configMu.RUnlock()

	return config
}

func maskSecret(s string) string {
	if s == "" {
		return ""
	}

	if len(s) <= 6 {
		return "******"
	}

	return s[:3] +
		"******" +
		s[len(s)-3:]
}

// ============================================================
// MAIN
// ============================================================

func main() {
	log.SetFlags(
		log.LstdFlags |
			log.Lmicroseconds,
	)

	fmt.Println()
	fmt.Println("========================================")
	fmt.Println(" QbotAI")
	fmt.Println(" Version:", AppVersion)
	fmt.Println("========================================")
	fmt.Println()

	if err := loadConfig(); err != nil {
		log.Fatal("配置文件错误:", err)
	}

	if err := ensureMemoryFile(); err != nil {
		log.Fatal("memory.md 初始化失败:", err)
	}

	aiBot = &AIClient{}

	cfg := getConfig()

	if cfg.QQ.Enabled {
		qqBot = NewQQBot()
		go qqBot.Run()
	}

	go startWebServer()

	fmt.Printf(
		"[Web] http://%s:%d\n",
		cfg.Web.Host,
		cfg.Web.Port,
	)

	fmt.Println(
		"[Memory]",
		MemoryFile,
	)

	fmt.Println()

	select {}
}

// ============================================================
// MEMORY
// ============================================================

func ensureMemoryFile() error {
	if _, err := os.Stat(MemoryFile); err == nil {
		return nil
	}

	content :=
		"# QbotAI Memory\n\n" +
			"> QbotAI 长期记忆文档。\n" +
			"> 本文件由 QbotAI 自动维护。\n\n"

	return os.WriteFile(
		MemoryFile,
		[]byte(content),
		0600,
	)
}

func readMemoryString() (string, error) {
	memoryMu.Lock()
	defer memoryMu.Unlock()

	data, err := os.ReadFile(MemoryFile)

	if err != nil {
		return "", err
	}

	return string(data), nil
}

func memorySize() int64 {
	info, err := os.Stat(MemoryFile)

	if err != nil {
		return 0
	}

	return info.Size()
}

func appendMemory(memory string) error {
	memory = strings.TrimSpace(memory)

	if memory == "" {
		return nil
	}

	memoryMu.Lock()
	defer memoryMu.Unlock()

	existing, err := os.ReadFile(MemoryFile)

	if err != nil {
		return err
	}

	now := time.Now().Format(
		"2006-01-02 15:04:05",
	)

	block :=
		"\n\n## " + now + "\n\n" +
			memory +
			"\n"

	newContent :=
		string(existing) +
			block

	tmp := MemoryFile + ".tmp"

	if err := os.WriteFile(
		tmp,
		[]byte(newContent),
		0600,
	); err != nil {
		return err
	}

	if err := os.Rename(
		tmp,
		MemoryFile,
	); err != nil {
		return err
	}

	return enforceMemoryLimitLocked()
}

func enforceMemoryLimitLocked() error {
	cfg := getConfig()

	maxBytes := cfg.Memory.MaxBytes

	if maxBytes <= 0 {
		maxBytes = MaxMemoryBytes
	}

	info, err := os.Stat(MemoryFile)

	if err != nil {
		return err
	}

	if info.Size() <= maxBytes {
		return nil
	}

	data, err := os.ReadFile(MemoryFile)

	if err != nil {
		return err
	}

	text := string(data)

	for len([]byte(text)) > int(maxBytes) {

		next, ok := removeOldestMemorySection(text)

		if !ok {
			break
		}

		text = next
	}

	tmp := MemoryFile + ".tmp"

	if err := os.WriteFile(
		tmp,
		[]byte(text),
		0600,
	); err != nil {
		return err
	}

	return os.Rename(
		tmp,
		MemoryFile,
	)
}

func removeOldestMemorySection(text string) (string, bool) {
	lines := strings.Split(
		text,
		"\n",
	)

	if len(lines) < 3 {
		return text, false
	}

	start := -1

	for i := 0; i < len(lines); i++ {
		if strings.HasPrefix(
			strings.TrimSpace(lines[i]),
			"## ",
		) {
			start = i
			break
		}
	}

	if start < 0 {
		return text, false
	}

	end := len(lines)

	for i := start + 1; i < len(lines); i++ {
		if strings.HasPrefix(
			strings.TrimSpace(lines[i]),
			"## ",
		) {
			end = i
			break
		}
	}

	result := append(
		[]string{},
		lines[:start]...,
	)

	result = append(
		result,
		lines[end:]...,
	)

	return strings.Join(
		result,
		"\n",
	), true
}

func compressMemoryIfNeeded() {
	cfg := getConfig()

	if !cfg.Memory.Enabled {
		return
	}

	every := cfg.Memory.CompressionEvery

	if every <= 0 {
		every = 15
	}

	count := atomic.LoadInt64(
		&aiOperationCounter,
	)

	if count < int64(every) {
		return
	}

	if !atomic.CompareAndSwapInt64(
		&aiOperationCounter,
		count,
		0,
	) {
		return
	}

	go compressMemory()
}

func compressMemory() {
	log.Println(
		"[Memory] 开始 AI 记忆压缩",
	)

	memory, err := readMemoryString()

	if err != nil {
		log.Println(
			"[Memory] 读取失败:",
			err,
		)
		return
	}

	if memory == "" {
		return
	}

	originalSize :=
		int64(len([]byte(memory)))

	targetRatio :=
		getConfig().Memory.CompressionRatio

	if targetRatio <= 0 ||
		targetRatio >= 1 {
		targetRatio = 0.75
	}

	targetSize :=
		int64(
			math.Max(
				1,
				float64(originalSize)*
					targetRatio,
			),
		)

	prompt := fmt.Sprintf(
		`你现在负责压缩 QbotAI 的长期 Markdown 记忆。

下面是完整的 memory.md。

你的任务：

1. 删除重复信息。
2. 删除没有长期价值的闲聊。
3. 合并表达相同的信息。
4. 保留人物、偏好、项目、重要事件、长期事实。
5. 保留日期结构。
6. 保持 Markdown 格式。
7. 不要解释压缩过程。
8. 只返回新的完整 Markdown 文档。
9. 当前大小约为 %d bytes。
10. 目标大小约为 %d bytes。
11. 尽量达到目标，但绝对不要为了减少大小而删除重要长期信息。

完整 memory.md：

%s`,
		originalSize,
		targetSize,
		memory,
	)

	result, err := aiBot.RawText(
		prompt,
	)

	if err != nil {
		log.Println(
			"[Memory] AI 压缩失败:",
			err,
		)
		return
	}

	result = cleanMarkdownResult(result)

	if strings.TrimSpace(result) == "" {
		return
	}

	newSize :=
		int64(len([]byte(result)))

	if newSize >= originalSize {
		log.Println(
			"[Memory] AI 压缩没有减小文件，保留原文件",
		)
		return
	}

	memoryMu.Lock()
	defer memoryMu.Unlock()

	tmp := MemoryFile + ".tmp"

	if err := os.WriteFile(
		tmp,
		[]byte(result),
		0600,
	); err != nil {
		log.Println(
			"[Memory] 写入失败:",
			err,
		)
		return
	}

	if err := os.Rename(
		tmp,
		MemoryFile,
	); err != nil {
		log.Println(
			"[Memory] 替换失败:",
			err,
		)
		return
	}

	_ = enforceMemoryLimitLocked()

	log.Printf(
		"[Memory] 压缩完成: %d -> %d bytes",
		originalSize,
		memorySize(),
	)
}

func cleanMarkdownResult(text string) string {
	text = strings.TrimSpace(text)

	if strings.HasPrefix(
		text,
		"```markdown",
	) {
		text = strings.TrimPrefix(
			text,
			"```markdown",
		)

		text = strings.TrimSpace(
			text,
		)
	}

	if strings.HasPrefix(
		text,
		"```md",
	) {
		text = strings.TrimPrefix(
			text,
			"```md",
		)

		text = strings.TrimSpace(
			text,
		)
	}

	if strings.HasPrefix(
		text,
		"```",
	) {
		text = strings.TrimPrefix(
			text,
			"```",
		)

		text = strings.TrimSpace(
			text,
		)
	}

	if strings.HasSuffix(
		text,
		"```",
	) {
		text = strings.TrimSuffix(
			text,
			"```",
		)

		text = strings.TrimSpace(
			text,
		)
	}

	return text
}

// ============================================================
// AI
// ============================================================

type AIClient struct{}

type AIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIRequest struct {
	Model       string      `json:"model"`
	Messages    []AIMessage `json:"messages"`
	Temperature float64     `json:"temperature,omitempty"`
	MaxTokens   int         `json:"max_tokens,omitempty"`
	Stream      bool        `json:"stream"`
}

type OpenAIResponse struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`

	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func normalizeBaseURL(v string) string {
	v = strings.TrimSpace(v)

	v = strings.TrimRight(
		v,
		"/",
	)

	if v == "" {
		v = "https://api.openai.com/v1"
	}

	return v
}

func (a *AIClient) request(
	messages []AIMessage,
) (string, error) {

	cfg := getConfig()

	if !cfg.AI.Enabled {
		return "", errors.New(
			"AI 功能未启用",
		)
	}

	if cfg.AI.APIKey == "" {
		return "", errors.New(
			"AI API Key 未配置",
		)
	}

	body := OpenAIRequest{
		Model: cfg.AI.Model,
		Messages: messages,

		Temperature: cfg.AI.Temperature,

		MaxTokens: cfg.AI.MaxTokens,

		Stream: false,
	}

	data, err := json.Marshal(body)

	if err != nil {
		return "", err
	}

	endpoint :=
		normalizeBaseURL(
			cfg.AI.BaseURL,
		) +
			"/chat/completions"

	req, err := http.NewRequest(
		http.MethodPost,
		endpoint,
		bytes.NewReader(data),
	)

	if err != nil {
		return "", err
	}

	req.Header.Set(
		"Authorization",
		"Bearer "+cfg.AI.APIKey,
	)

	req.Header.Set(
		"Content-Type",
		"application/json",
	)

	req.Header.Set(
		"User-Agent",
		AppName+"/"+AppVersion,
	)

	client := &http.Client{
		Timeout: 180 * time.Second,
	}

	resp, err := client.Do(req)

	if err != nil {
		return "", err
	}

	defer resp.Body.Close()

	responseBody, err :=
		io.ReadAll(
			io.LimitReader(
				resp.Body,
				64*1024*1024,
			),
		)

	if err != nil {
		return "", err
	}

	if resp.StatusCode < 200 ||
		resp.StatusCode >= 300 {

		var e OpenAIResponse

		if json.Unmarshal(
			responseBody,
			&e,
		) == nil &&
			e.Error != nil {

			return "",
				fmt.Errorf(
					"AI HTTP %d: %s",
					resp.StatusCode,
					e.Error.Message,
				)
		}

		return "",
			fmt.Errorf(
				"AI HTTP %d: %s",
				resp.StatusCode,
				strings.TrimSpace(
					string(responseBody),
				),
			)
	}

	var result OpenAIResponse

	if err := json.Unmarshal(
		responseBody,
		&result,
	); err != nil {
		return "",
			fmt.Errorf(
				"AI JSON 解析失败: %w",
				err,
			)
	}

	if result.Error != nil {
		return "",
			errors.New(
				result.Error.Message,
			)
	}

	if len(result.Choices) == 0 {
		return "",
			errors.New(
				"AI 没有返回 choices",
			)
	}

	return strings.TrimSpace(
		result.Choices[0].Message.Content,
	), nil
}

func (a *AIClient) RawText(
	prompt string,
) (string, error) {

	cfg := getConfig()

	memory, err := readMemoryString()

	if err != nil {
		memory = ""
	}

	messages := []AIMessage{
		{
			Role: "system",
			Content: cfg.AI.SystemPrompt,
		},
		{
			Role: "user",
			Content:
				"【QbotAI长期记忆 memory.md】\n\n" +
					memory +
					"\n\n" +
					"【任务】\n\n" +
					prompt,
		},
	}

	atomic.AddInt64(
		&aiOperationCounter,
		1,
	)

	return a.request(messages)
}

type AIDecision struct {
	Reply bool `json:"reply"`

	Message string `json:"message"`

	Memory []string `json:"memory"`
}

func (a *AIClient) DecideGroupReply(
	groupID string,
	current string,
	forced bool,
) (AIDecision, error) {

	cfg := getConfig()

	memory := ""

	if cfg.AI.MemoryEnabled &&
		cfg.Memory.Enabled {

		var err error

		memory, err =
			readMemoryString()

		if err != nil {
			log.Println(
				"[Memory] 读取失败:",
				err,
			)
		}
	}

	context := getGroupContext(
		groupID,
	)

	contextText :=
		context.String()

	mode := "普通群聊"

	if forced {
		mode = "用户明确 @ QbotAI"
	}

	prompt := fmt.Sprintf(
		`你正在作为一个真实QQ群成员参与群聊。

当前模式：%s

你需要判断 QbotAI 现在是否应该说话。

重要规则：

1. 普通群聊不要每句话都回复。
2. 没有必要说话时 reply 必须是 false。
3. 如果能够帮助群友、自然接话、回答问题、补充有价值的信息，可以 reply=true。
4. 不要为了刷存在感发言。
5. 如果只是“哈哈”“好的”“哦”“嗯”等没有价值的消息，一般不要回复。
6. 如果群聊正在自然进行，而且没有必要插话，不要回复。
7. 如果有人明显需要回答，而群里没人回答，可以回复。
8. 如果用户明确 @ QbotAI，必须 reply=true。
9. message 必须是实际准备发送到QQ群里的内容。
10. memory 只记录长期有价值的信息，不要记录普通闲聊。
11. 不要把当前一次性对话全部写进 memory。
12. memory 中的内容如果与已有长期记忆重复，不要再次添加。

你的输出必须严格是 JSON：

{
  "reply": true,
  "message": "要发送的内容",
  "memory": [
    "值得长期保存的信息"
  ]
}

如果不应该回复：

{
  "reply": false,
  "message": "",
  "memory": []
}

当前 QbotAI 主动程度：%d/100

最近群聊：

%s

当前消息：

%s

完整长期记忆 memory.md：

%s`,
		mode,
		cfg.GroupAI.ProactiveLevel,
		contextText,
		current,
		memory,
	)

	system :=
		cfg.AI.SystemPrompt +
			"\n\n你必须遵守输出 JSON 的要求。" +
			"\n你是QQ群里的自然成员，不是客服机器人。"

	messages := []AIMessage{
		{
			Role:    "system",
			Content: system,
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	atomic.AddInt64(
		&aiOperationCounter,
		1,
	)

	raw, err := a.request(messages)

	if err != nil {
		return AIDecision{}, err
	}

	decision, err :=
		parseAIDecision(raw)

	if err != nil {
		return AIDecision{}, err
	}

	if forced {
		decision.Reply = true
	}

	return decision, nil
}

func parseAIDecision(
	raw string,
) (AIDecision, error) {

	raw = strings.TrimSpace(raw)

	raw = cleanJSONCodeFence(raw)

	var result AIDecision

	if err := json.Unmarshal(
		[]byte(raw),
		&result,
	); err != nil {

		start := strings.Index(
			raw,
			"{",
		)

		end := strings.LastIndex(
			raw,
			"}",
		)

		if start >= 0 &&
			end > start {

			raw2 :=
				raw[start : end+1]

			if err2 := json.Unmarshal(
				[]byte(raw2),
				&result,
			); err2 == nil {

				return result, nil
			}
		}

		return result,
			fmt.Errorf(
				"AI 没有返回合法 JSON: %s",
				raw,
			)
	}

	return result, nil
}

func cleanJSONCodeFence(
	s string,
) string {

	s = strings.TrimSpace(s)

	s = strings.TrimPrefix(
		s,
		"```json",
	)

	s = strings.TrimPrefix(
		s,
		"```JSON",
	)

	s = strings.TrimPrefix(
		s,
		"```",
	)

	s = strings.TrimSuffix(
		s,
		"```",
	)

	return strings.TrimSpace(s)
}

// ============================================================
// GROUP CONTEXT
// ============================================================

type ChatMessage struct {
	UserID    string
	MessageID string
	Time      time.Time
	Content   string
}

type GroupContext struct {
	Messages []ChatMessage
}

func getGroupContext(
	groupID string,
) *GroupContext {

	groupMu.Lock()
	defer groupMu.Unlock()

	ctx := groupContexts[groupID]

	if ctx == nil {
		ctx = &GroupContext{}
		groupContexts[groupID] = ctx
	}

	return ctx
}

func addGroupMessage(
	groupID string,
	msg ChatMessage,
) {

	cfg := getConfig()

	groupMu.Lock()
	defer groupMu.Unlock()

	ctx := groupContexts[groupID]

	if ctx == nil {
		ctx = &GroupContext{}
		groupContexts[groupID] = ctx
	}

	ctx.Messages =
		append(
			ctx.Messages,
			msg,
		)

	max := cfg.GroupAI.ContextMessages

	if max <= 0 {
		max = 20
	}

	if len(ctx.Messages) > max {
		ctx.Messages =
			ctx.Messages[
				len(ctx.Messages)-max:
			]
	}
}

func (c *GroupContext) String() string {

	if c == nil ||
		len(c.Messages) == 0 {

		return "(暂无历史消息)"
	}

	var b strings.Builder

	for _, msg := range c.Messages {

		b.WriteString(
			"[",
		)

		b.WriteString(
			msg.Time.Format(
				"15:04:05",
			),
		)

		b.WriteString(
			"] 用户",
		)

		b.WriteString(
			msg.UserID,
		)

		b.WriteString(
			"：",
		)

		b.WriteString(
			msg.Content,
		)

		b.WriteString(
			"\n",
		)
	}

	return b.String()
}

// ============================================================
// QQ BOT TOKEN
// ============================================================

type TokenManager struct {
	mu sync.Mutex

	accessToken string

	expiresAt time.Time
}

func (t *TokenManager) Get() (
	string,
	error,
) {

	t.mu.Lock()
	defer t.mu.Unlock()

	if t.accessToken != "" &&
		time.Now().Before(
			t.expiresAt.Add(
				-5*time.Minute,
			),
		) {

		return t.accessToken, nil
	}

	cfg := getConfig()

	if cfg.QQ.AppID == "" {
		return "",
			errors.New(
				"QQ AppID 未配置",
			)
	}

	if cfg.QQ.ClientSecret == "" {
		return "",
			errors.New(
				"QQ Client Secret 未配置",
			)
	}

	payload := map[string]string{
		"appId":        cfg.QQ.AppID,
		"clientSecret": cfg.QQ.ClientSecret,
	}

	data, err :=
		json.Marshal(payload)

	if err != nil {
		return "", err
	}

	req, err :=
		http.NewRequest(
			http.MethodPost,
			QQTokenBase+
				"/app/getAppAccessToken",
			bytes.NewReader(data),
		)

	if err != nil {
		return "", err
	}

	req.Header.Set(
		"Content-Type",
		"application/json",
	)

	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err :=
		client.Do(req)

	if err != nil {
		return "", err
	}

	defer resp.Body.Close()

	body, err :=
		io.ReadAll(
			io.LimitReader(
				resp.Body,
				2*1024*1024,
			),
		)

	if err != nil {
		return "", err
	}

	if resp.StatusCode < 200 ||
		resp.StatusCode >= 300 {

		return "",
			fmt.Errorf(
				"QQ Token HTTP %d: %s",
				resp.StatusCode,
				string(body),
			)
	}

	var result struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   Int64Flexible  `json:"expires_in"`
	}

	if err := json.Unmarshal(
		body,
		&result,
	); err != nil {
		return "", err
	}

	if result.AccessToken == "" {
		return "",
			errors.New(
				"QQ 没有返回 access_token",
			)
	}

	t.accessToken =
		result.AccessToken

	t.expiresAt =
		time.Now().Add(
			time.Duration(
				result.ExpiresIn,
			) * time.Second,
		)

	log.Println(
		"[QQ] AccessToken 获取成功",
	)

	return t.accessToken, nil
}

// ============================================================
// QQ BOT
// ============================================================

type QQBot struct {
	token TokenManager

	mu sync.Mutex

	conn *WSConn

	seq int64

	sessionID string

	resumeURL string

	connected atomic.Bool

	msgSeq uint32
}

func NewQQBot() *QQBot {
	return &QQBot{}
}

func (q *QQBot) Run() {

	for {

		cfg := getConfig()

		if !cfg.QQ.Enabled {
			time.Sleep(
				3 * time.Second,
			)

			continue
		}

		log.Println(
			"[QQ] 正在连接 Gateway...",
		)

		err :=
			q.connectAndRun()

		if err != nil {
			log.Println(
				"[QQ] Gateway:",
				err,
			)
		}

		q.connected.Store(false)

		time.Sleep(
			5 * time.Second,
		)
	}
}

func (q *QQBot) getGateway() (
	string,
	error,
) {

	token, err :=
		q.token.Get()

	if err != nil {
		return "", err
	}

	req, err :=
		http.NewRequest(
			http.MethodGet,
			QQAPIBase+
				"/gateway",
			nil,
		)

	if err != nil {
		return "", err
	}

	req.Header.Set(
		"Authorization",
		"QQBot "+token,
	)

	req.Header.Set(
		"User-Agent",
		AppName+"/"+AppVersion,
	)

	resp, err :=
		http.DefaultClient.Do(req)

	if err != nil {
		return "", err
	}

	defer resp.Body.Close()

	body, err :=
		io.ReadAll(
			io.LimitReader(
				resp.Body,
				2*1024*1024,
			),
		)

	if err != nil {
		return "", err
	}

	if resp.StatusCode < 200 ||
		resp.StatusCode >= 300 {

		return "",
			fmt.Errorf(
				"Gateway HTTP %d: %s",
				resp.StatusCode,
				string(body),
			)
	}

	var result struct {
		URL string `json:"url"`
	}

	if err := json.Unmarshal(
		body,
		&result,
	); err != nil {
		return "", err
	}

	if result.URL == "" {
		return "",
			errors.New(
				"Gateway URL 为空",
			)
	}

	return result.URL, nil
}

func (q *QQBot) connectAndRun() error {

	gateway, err :=
		q.getGateway()

	if err != nil {
		return err
	}

	conn, err :=
		DialWebSocket(gateway)

	if err != nil {
		return err
	}

	q.mu.Lock()

	q.conn = conn

	q.mu.Unlock()

	defer func() {

		conn.Close()

		q.mu.Lock()

		q.conn = nil

		q.mu.Unlock()

		q.connected.Store(false)
	}()

	_, helloData, err :=
		conn.ReadMessage()

	if err != nil {
		return err
	}

	var hello QQGatewayEvent

	if err := json.Unmarshal(
		helloData,
		&hello,
	); err != nil {
		return err
	}

	if hello.Op != 10 {
		return fmt.Errorf(
			"QQ Gateway HELLO 错误: OP=%d",
			hello.Op,
		)
	}

	var helloInfo struct {
		HeartbeatInterval int `json:"heartbeat_interval"`
	}

	if err := json.Unmarshal(
		hello.D,
		&helloInfo,
	); err != nil {
		return err
	}

	if err := q.identify(); err != nil {
		return err
	}

	q.connected.Store(true)

	go q.heartbeatLoop(
		time.Duration(
			helloInfo.HeartbeatInterval,
		) * time.Millisecond,
	)

	for {

		_, data, err :=
			conn.ReadMessage()

		if err != nil {
			return err
		}

		var event QQGatewayEvent

		if err := json.Unmarshal(
			data,
			&event,
		); err != nil {
			continue
		}

		if event.S != 0 {
			q.seq = event.S
		}

		switch event.Op {

		case 0:
			q.handleDispatch(
				&event,
			)

		case 1:
			_ = q.sendHeartbeat()

		case 7:
			return errors.New(
				"Gateway 要求重新连接",
			)

		case 9:
			return errors.New(
				"Gateway Invalid Session",
			)

		case 11:
			log.Println(
				"[QQ] Heartbeat ACK",
			)
		}
	}
}

type QQGatewayEvent struct {
	Op int             `json:"op"`
	D  json.RawMessage `json:"d"`
	S  int64           `json:"s"`
	T string          `json:"t"`
}

func (q *QQBot) identify() error {

	token, err :=
		q.token.Get()

	if err != nil {
		return err
	}

	payload := map[string]interface{}{
		"op": 2,
		"d": map[string]interface{}{
			"token":
				"QQBot " + token,

			"intents":
				GroupMessageIntent,

			"shard": []int{
				0,
				1,
			},

			"properties": map[string]string{
				"$os":      "QbotAI",
				"$browser": "QbotAI",
				"$device":  "QbotAI",
			},
		},
	}

	return q.sendJSON(
		payload,
	)
}

func (q *QQBot) sendJSON(
	v interface{},
) error {

	data, err :=
		json.Marshal(v)

	if err != nil {
		return err
	}

	q.mu.Lock()

	defer q.mu.Unlock()

	if q.conn == nil {
		return errors.New(
			"QQ Gateway 未连接",
		)
	}

	return q.conn.WriteMessage(
		data,
	)
}

func (q *QQBot) sendHeartbeat() error {

	payload := map[string]interface{}{
		"op": 1,
		"d":  q.seq,
	}

	return q.sendJSON(
		payload,
	)
}

func (q *QQBot) heartbeatLoop(
	interval time.Duration,
) {

	if interval <= 0 {
		interval =
			30 * time.Second
	}

	ticker :=
		time.NewTicker(
			interval,
		)

	defer ticker.Stop()

	for range ticker.C {

		if !q.connected.Load() {
			return
		}

		_ = q.sendHeartbeat()
	}
}

func (q *QQBot) handleDispatch(
	event *QQGatewayEvent,
) {

	switch event.T {

	case "READY":
		var d struct {
			SessionID string `json:"session_id"`
			ResumeURL string `json:"resume_gateway_url"`
		}

		if json.Unmarshal(
			event.D,
			&d,
		) == nil {

			q.sessionID =
				d.SessionID

			q.resumeURL =
				d.ResumeURL
		}

		log.Println(
			"[QQ] READY，机器人上线",
		)

	case "GROUP_MESSAGE_CREATE":
		q.handleGroupMessage(
			event.D,
		)

	case "GROUP_AT_MESSAGE_CREATE":
		q.handleGroupMessage(
			event.D,
		)

	case "RESUMED":
		log.Println(
			"[QQ] Session RESUMED",
		)

	default:
		log.Printf(
			"[QQ] Event: %s\n",
			event.T,
		)
	}
}

// ============================================================
// QQ GROUP MESSAGE
// ============================================================

type QQGroupMessage struct {
	ID string `json:"id"`

	Content string `json:"content"`

	GroupOpenID string `json:"group_openid"`

	Timestamp string `json:"timestamp"`

	Author struct {
		ID string `json:"id"`

		MemberOpenID string `json:"member_openid"`
	} `json:"author"`

	Mentions []struct {
		ID string `json:"id"`

		MemberOpenID string `json:"member_openid"`
	} `json:"mentions"`
}

func (q *QQBot) handleGroupMessage(
	data []byte,
) {

	var msg QQGroupMessage

	if err := json.Unmarshal(
		data,
		&msg,
	); err != nil {

		log.Println(
			"[QQ] 群消息 JSON 错误:",
			err,
		)

		return
	}

	content :=
		strings.TrimSpace(
			msg.Content,
		)

	if content == "" {
		return
	}

	userID :=
		msg.Author.MemberOpenID

	if userID == "" {
		userID =
			msg.Author.ID
	}

	mentioned :=
		q.isMentioned(
			&msg,
		)

	content =
		removeBotMention(
			content,
			mentioned,
		)

	now := time.Now()

	addGroupMessage(
		msg.GroupOpenID,
		ChatMessage{
			UserID:    userID,
			MessageID: msg.ID,
			Time:      now,
			Content:   content,
		},
	)

	log.Printf(
		"[QQ] 群=%s 用户=%s @=%v 内容=%s",
		msg.GroupOpenID,
		userID,
		mentioned,
		content,
	)

	if strings.HasPrefix(
		content,
		"/",
	) {

		if q.handleCommand(
			msg.GroupOpenID,
			msg.ID,
			content,
		) {

			return
		}
	}

	if mentioned {

		go q.aiReply(
			msg.GroupOpenID,
			msg.ID,
			content,
			true,
		)

		return
	}

	cfg := getConfig()

	if !cfg.GroupAI.Enabled ||
		!cfg.AI.Enabled {

		return
	}

	delay :=
		time.Duration(
			cfg.GroupAI.DecisionDelaySeconds,
		) * time.Second

	if delay <= 0 {
		delay = 3 * time.Second
	}

	go func() {

		time.Sleep(delay)

		if q.shouldSkipDueToNewMessages(
			msg.GroupOpenID,
			msg.ID,
		) {
			return
		}

		q.aiReply(
			msg.GroupOpenID,
			msg.ID,
			content,
			false,
		)
	}()
}

func (q *QQBot) isMentioned(
	msg *QQGroupMessage,
) bool {

	for _, mention :=
		range msg.Mentions {

		if mention.ID != "" {
			return true
		}

		if mention.MemberOpenID != "" {
			return true
		}
	}

	return strings.Contains(
		msg.Content,
		"<@",
	)
}

func removeBotMention(
	content string,
	mentioned bool,
) string {

	if !mentioned {
		return content
	}

	for {

		start :=
			strings.Index(
				content,
				"<@",
			)

		if start < 0 {
			break
		}

		endRelative :=
			strings.Index(
				content[start:],
				">",
			)

		if endRelative < 0 {
			break
		}

		end :=
			start +
				endRelative +
				1

		content =
			strings.TrimSpace(
				content[:start]+
					content[end:],
			)
	}

	return content
}

func (q *QQBot) shouldSkipDueToNewMessages(
	groupID string,
	messageID string,
) bool {

	groupMu.Lock()
	defer groupMu.Unlock()

	ctx := groupContexts[groupID]

	if ctx == nil {
		return false
	}

	for _, m := range ctx.Messages {

		if m.MessageID == messageID {
			continue
		}

		if m.Time.After(
			startTime,
		) &&
			m.MessageID != messageID {

			if m.Time.After(
				time.Now().Add(
					-10*time.Second,
				),
			) {
				return false
			}
		}
	}

	return false
}

// ============================================================
// AI GROUP REPLY
// ============================================================

func (q *QQBot) aiReply(
	groupID string,
	messageID string,
	content string,
	forced bool,
) {

	cfg := getConfig()

	if !cfg.AI.Enabled {
		return
	}

	if !forced {

		if !q.canReply(groupID) {
			return
		}
	}

	decision, err :=
		aiBot.DecideGroupReply(
			groupID,
			content,
			forced,
		)

	if err != nil {

		log.Println(
			"[AI] 群聊判断失败:",
			err,
		)

		return
	}

	if len(decision.Memory) > 0 &&
		cfg.Memory.Enabled {

		for _, memory :=
			range decision.Memory {

			memory =
				strings.TrimSpace(
					memory,
				)

			if memory == "" {
				continue
			}

			if err := appendMemory(
				memory,
			); err != nil {

				log.Println(
					"[Memory] 写入失败:",
					err,
				)

				continue
			}

			atomic.AddInt64(
				&aiOperationCounter,
				1,
			)
		}
	}

	compressMemoryIfNeeded()

	if !decision.Reply {
		return
	}

	answer :=
		strings.TrimSpace(
			decision.Message,
		)

	if answer == "" {
		return
	}

	if !forced &&
		!q.canReply(groupID) {

		return
	}

	if err := q.sendGroupMessage(
		groupID,
		answer,
		messageID,
	); err != nil {

		log.Println(
			"[QQ] 发送失败:",
			err,
		)

		return
	}

	if !forced {
		recordReply(groupID)
	}
}

// ============================================================
// REPLY LIMIT
// ============================================================

func (q *QQBot) canReply(
	groupID string,
) bool {

	cfg := getConfig()

	replyMu.Lock()
	defer replyMu.Unlock()

	now := time.Now()

	list :=
		replyHistory[groupID]

	cutoff :=
		now.Add(
			-1 * time.Hour,
		)

	filtered :=
		make([]time.Time, 0)

	for _, t := range list {

		if t.After(cutoff) {
			filtered =
				append(
					filtered,
					t,
				)
		}
	}

	replyHistory[groupID] =
		filtered

	if len(filtered) >=
		cfg.GroupAI.MaxRepliesPerHour {

		return false
	}

	if len(filtered) > 0 {

		last :=
			filtered[
				len(filtered)-1,
			]

		cooldown :=
			time.Duration(
				cfg.GroupAI.ReplyCooldownSeconds,
			) * time.Second

		if now.Sub(last) <
			cooldown {

			return false
		}
	}

	return true
}

func recordReply(
	groupID string,
) {

	replyMu.Lock()
	defer replyMu.Unlock()

	replyHistory[groupID] =
		append(
			replyHistory[groupID],
			time.Now(),
		)
}

// ============================================================
// QQ COMMANDS
// ============================================================

func (q *QQBot) handleCommand(
	groupID string,
	messageID string,
	content string,
) bool {

	fields :=
		strings.Fields(content)

	if len(fields) == 0 {
		return false
	}

	command :=
		strings.ToLower(
			fields[0],
		)

	switch command {

	case "/menu":

		q.sendGroupMessage(
			groupID,
			"QbotAI 功能菜单\n\n"+
				"/menu - 查看菜单\n"+
				"/ai 内容 - 直接调用 AI\n"+
				"/model - 当前模型\n"+
				"/prompt - 当前提示词\n"+
				"/memory - 查看记忆状态\n"+
				"/about - QbotAI 信息",
			messageID,
		)

		return true

	case "/about":

		cfg := getConfig()

		q.sendGroupMessage(
			groupID,
			"QbotAI\n"+
				"Version: "+AppVersion+"\n"+
				"Model: "+cfg.AI.Model+"\n"+
				"Memory: "+formatBytes(
					memorySize(),
				),
			messageID,
		)

		return true

	case "/model":

		cfg := getConfig()

		q.sendGroupMessage(
			groupID,
			"当前 AI 模型：\n"+
				cfg.AI.Model,
			messageID,
		)

		return true

	case "/prompt":

		cfg := getConfig()

		prompt :=
			cfg.AI.SystemPrompt

		if len([]rune(prompt)) > 1500 {
			prompt =
				string(
					[]rune(prompt)[:1500],
				)
		}

		q.sendGroupMessage(
			groupID,
			"当前 System Prompt：\n"+
				prompt,
			messageID,
		)

		return true

	case "/memory":

		cfg := getConfig()

		count :=
			atomic.LoadInt64(
				&aiOperationCounter,
			)

		q.sendGroupMessage(
			groupID,
			fmt.Sprintf(
				"QbotAI Memory\n\n"+
					"文件：memory.md\n"+
					"大小：%s / %s\n"+
					"压缩计数：%d / %d\n"+
					"压缩目标：%.0f%%",
				formatBytes(
					memorySize(),
				),
				formatBytes(
					cfg.Memory.MaxBytes,
				),
				count,
				cfg.Memory.CompressionEvery,
				cfg.Memory.CompressionRatio*
					100,
			),
			messageID,
		)

		return true

	case "/ai":

		text :=
			strings.TrimSpace(
				strings.TrimPrefix(
					content,
					fields[0],
				),
			)

		if text == "" {

			q.sendGroupMessage(
				groupID,
				"用法：/ai 你的问题",
				messageID,
			)

			return true
		}

		go func() {

			answer, err :=
				aiBot.RawText(text)

			if err != nil {

				q.sendGroupMessage(
					groupID,
					"AI 请求失败："+
						err.Error(),
					messageID,
				)

				return
			}

			q.sendGroupMessage(
				groupID,
				answer,
				messageID,
			)
		}()

		return true
	}

	return false
}

// ============================================================
// QQ SEND MESSAGE
// ============================================================

func (q *QQBot) sendGroupMessage(
	groupID string,
	content string,
	replyTo string,
) error {

	token, err :=
		q.token.Get()

	if err != nil {
		return err
	}

	content =
		strings.TrimSpace(
			content,
		)

	if content == "" {
		return nil
	}

	// QQ 文本消息长度控制。
	runes :=
		[]rune(content)

	if len(runes) > 4500 {
		content =
			string(runes[:4500])
	}

	seq :=
		atomic.AddUint32(
			&q.msgSeq,
			1,
		)

	if seq == 0 {
		seq = 1
	}

	body := map[string]interface{}{
		"content": content,
		"msg_type": 0,
		"msg_seq":  seq,
	}

	if replyTo != "" {
		body["msg_id"] = replyTo
	}

	data, err :=
		json.Marshal(body)

	if err != nil {
		return err
	}

	endpoint :=
		QQAPIBase +
			"/v2/groups/" +
			url.PathEscape(groupID) +
			"/messages"

	req, err :=
		http.NewRequest(
			http.MethodPost,
			endpoint,
			bytes.NewReader(data),
		)

	if err != nil {
		return err
	}

	req.Header.Set(
		"Authorization",
		"QQBot "+token,
	)

	req.Header.Set(
		"Content-Type",
		"application/json",
	)

	req.Header.Set(
		"User-Agent",
		AppName+"/"+AppVersion,
	)

	resp, err :=
		http.DefaultClient.Do(req)

	if err != nil {
		return err
	}

	defer resp.Body.Close()

	bodyResp, _ :=
		io.ReadAll(
			io.LimitReader(
				resp.Body,
				4*1024*1024,
			),
		)

	if resp.StatusCode < 200 ||
		resp.StatusCode >= 300 {

		return fmt.Errorf(
			"QQ 发送消息 HTTP %d: %s",
			resp.StatusCode,
			string(bodyResp),
		)
	}

	return nil
}

// ============================================================
// WEB SERVER
// ============================================================

func startWebServer() {

	mux := http.NewServeMux()

	mux.HandleFunc(
		"/",
		webIndex,
	)

	mux.HandleFunc(
		"/api/config",
		apiConfig,
	)

	mux.HandleFunc(
		"/api/save",
		apiSave,
	)

	mux.HandleFunc(
		"/api/status",
		apiStatus,
	)

	mux.HandleFunc(
		"/api/test-ai",
		apiTestAI,
	)

	mux.HandleFunc(
		"/api/memory",
		apiMemory,
	)

	mux.HandleFunc(
		"/api/compress-memory",
		apiCompressMemory,
	)

	cfg := getConfig()

	addr :=
		cfg.Web.Host +
			":" +
			strconv.Itoa(
				cfg.Web.Port,
			)

	server :=
		&http.Server{
			Addr: addr,

			Handler:
				logMiddleware(
					mux,
				),

			ReadTimeout:
				30 * time.Second,

			WriteTimeout:
				180 * time.Second,

			IdleTimeout:
				60 * time.Second,
		}

	log.Println(
		"[Web] 管理后台:",
		"http://"+addr,
	)

	if err := server.ListenAndServe(); err != nil &&
		err != http.ErrServerClosed {

		log.Println(
			"[Web]",
			err,
		)
	}
}

func logMiddleware(
	next http.Handler,
) http.Handler {

	return http.HandlerFunc(
		func(
			w http.ResponseWriter,
			r *http.Request,
		) {

			log.Printf(
				"[WEB] %s %s",
				r.Method,
				r.URL.Path,
			)

			next.ServeHTTP(
				w,
				r,
			)
		},
	)
}

func jsonResponse(
	w http.ResponseWriter,
	status int,
	v interface{},
) {

	w.Header().Set(
		"Content-Type",
		"application/json; charset=utf-8",
	)

	w.WriteHeader(status)

	_ = json.NewEncoder(w).Encode(v)
}

func apiConfig(
	w http.ResponseWriter,
	r *http.Request,
) {

	if r.Method != http.MethodGet {

		jsonResponse(
			w,
			405,
			map[string]string{
				"error":
					"method not allowed",
			},
		)

		return
	}

	cfg := getConfig()

	result :=
		map[string]interface{}{
			"qq": map[string]interface{}{
				"app_id":
					cfg.QQ.AppID,

				"client_secret":
					maskSecret(
						cfg.QQ.ClientSecret,
					),

				"enabled":
					cfg.QQ.Enabled,
			},

			"ai": map[string]interface{}{
				"enabled":
					cfg.AI.Enabled,

				"base_url":
					cfg.AI.BaseURL,

				"api_key":
					maskSecret(
						cfg.AI.APIKey,
					),

				"model":
					cfg.AI.Model,

				"system_prompt":
					cfg.AI.SystemPrompt,

				"temperature":
					cfg.AI.Temperature,

				"max_tokens":
					cfg.AI.MaxTokens,

				"memory_enabled":
					cfg.AI.MemoryEnabled,
			},

			"group_ai":
				cfg.GroupAI,

			"memory":
				cfg.Memory,

			"web":
				cfg.Web,

			"bot":
				cfg.Bot,
		}

	jsonResponse(
		w,
		200,
		result,
	)
}

type SaveRequest struct {
	QQ struct {
		AppID        string `json:"app_id"`
		ClientSecret string `json:"client_secret"`
		Enabled      bool   `json:"enabled"`
	} `json:"qq"`

	AI struct {
		Enabled       bool    `json:"enabled"`
		BaseURL       string  `json:"base_url"`
		APIKey        string  `json:"api_key"`
		Model         string  `json:"model"`
		SystemPrompt  string  `json:"system_prompt"`
		Temperature   float64 `json:"temperature"`
		MaxTokens     int     `json:"max_tokens"`
		MemoryEnabled bool    `json:"memory_enabled"`
	} `json:"ai"`

	GroupAI struct {
		Enabled bool `json:"enabled"`

		ContextMessages int `json:"context_messages"`

		DecisionDelaySeconds int `json:"decision_delay_seconds"`

		ReplyCooldownSeconds int `json:"reply_cooldown_seconds"`

		MaxRepliesPerHour int `json:"max_replies_per_hour"`

		ProactiveLevel int `json:"proactive_level"`

		OnlyReplyWhenUseful bool `json:"only_reply_when_useful"`
	} `json:"group_ai"`

	Memory struct {
		Enabled bool `json:"enabled"`

		CompressionEvery int `json:"compression_every"`

		CompressionRatio float64 `json:"compression_ratio"`

		MaxBytes int64 `json:"max_bytes"`

		DeleteOldestFirst bool `json:"delete_oldest_first"`
	} `json:"memory"`

	Web struct {
		Host string `json:"host"`
		Port int    `json:"port"`
	} `json:"web"`

	Bot struct {
		Name   string `json:"name"`
		Prefix string `json:"prefix"`
	} `json:"bot"`
}

func apiSave(
	w http.ResponseWriter,
	r *http.Request,
) {

	if r.Method != http.MethodPost {

		jsonResponse(
			w,
			405,
			map[string]string{
				"error":
					"method not allowed",
			},
		)

		return
	}

	var req SaveRequest

	if err := json.NewDecoder(
		r.Body,
	).Decode(&req); err != nil {

		jsonResponse(
			w,
			400,
			map[string]string{
				"error":
					"JSON 格式错误",
			},
		)

		return
	}

	configMu.Lock()

	config.QQ.AppID =
		strings.TrimSpace(
			req.QQ.AppID,
		)

	if req.QQ.ClientSecret != "" &&
		req.QQ.ClientSecret != "******" {

		config.QQ.ClientSecret =
			strings.TrimSpace(
				req.QQ.ClientSecret,
			)
	}

	config.QQ.Enabled =
		req.QQ.Enabled

	config.AI.Enabled =
		req.AI.Enabled

	if req.AI.BaseURL != "" {

		config.AI.BaseURL =
			normalizeBaseURL(
				req.AI.BaseURL,
			)
	}

	if req.AI.APIKey != "" &&
		req.AI.APIKey != "******" {

		config.AI.APIKey =
			strings.TrimSpace(
				req.AI.APIKey,
			)
	}

	if req.AI.Model != "" {

		config.AI.Model =
			strings.TrimSpace(
				req.AI.Model,
			)
	}

	config.AI.SystemPrompt =
		req.AI.SystemPrompt

	if req.AI.Temperature >= 0 &&
		req.AI.Temperature <= 2 {

		config.AI.Temperature =
			req.AI.Temperature
	}

	if req.AI.MaxTokens > 0 {

		config.AI.MaxTokens =
			req.AI.MaxTokens
	}

	config.AI.MemoryEnabled =
		req.AI.MemoryEnabled

	config.GroupAI =
		req.GroupAI

	if config.GroupAI.ContextMessages <= 0 {
		config.GroupAI.ContextMessages = 20
	}

	if config.GroupAI.DecisionDelaySeconds < 0 {
		config.GroupAI.DecisionDelaySeconds = 0
	}

	if config.GroupAI.ReplyCooldownSeconds < 0 {
		config.GroupAI.ReplyCooldownSeconds = 0
	}

	if config.GroupAI.MaxRepliesPerHour <= 0 {
		config.GroupAI.MaxRepliesPerHour = 30
	}

	config.GroupAI.ProactiveLevel =
		int(
			math.Max(
				0,
				math.Min(
					100,
					float64(
						config.GroupAI.ProactiveLevel,
					),
				),
			),
		)

	config.Memory =
		req.Memory

	if config.Memory.CompressionEvery <= 0 {
		config.Memory.CompressionEvery = 15
	}

	if config.Memory.CompressionRatio <= 0 ||
		config.Memory.CompressionRatio >= 1 {

		config.Memory.CompressionRatio =
			0.75
	}

	if config.Memory.MaxBytes <= 0 ||
		config.Memory.MaxBytes >
			MaxMemoryBytes {

		config.Memory.MaxBytes =
			MaxMemoryBytes
	}

	if req.Web.Host != "" {
		config.Web.Host =
			req.Web.Host
	}

	if req.Web.Port >= 1 &&
		req.Web.Port <= 65535 {

		config.Web.Port =
			req.Web.Port
	}

	if req.Bot.Name != "" {
		config.Bot.Name =
			req.Bot.Name
	}

	if req.Bot.Prefix != "" {
		config.Bot.Prefix =
			req.Bot.Prefix
	}

	err := saveConfigLocked()

	configMu.Unlock()

	if err != nil {

		jsonResponse(
			w,
			500,
			map[string]string{
				"error": err.Error(),
			},
		)

		return
	}

	_ = enforceMemoryLimit()

	jsonResponse(
		w,
		200,
		map[string]interface{}{
			"success": true,
			"message":
				"配置保存成功",
		},
	)
}

func enforceMemoryLimit() error {

	memoryMu.Lock()
	defer memoryMu.Unlock()

	return enforceMemoryLimitLocked()
}

func apiStatus(
	w http.ResponseWriter,
	r *http.Request,
) {

	cfg := getConfig()

	qqOnline := false

	if qqBot != nil {
		qqOnline =
			qqBot.connected.Load()
	}

	jsonResponse(
		w,
		200,
		map[string]interface{}{
			"name":
				AppName,

			"version":
				AppVersion,

			"uptime":
				time.Since(
					startTime,
				).String(),

			"qq_enabled":
				cfg.QQ.Enabled,

			"qq_online":
				qqOnline,

			"ai_enabled":
				cfg.AI.Enabled,

			"model":
				cfg.AI.Model,

			"memory_size":
				memorySize(),

			"memory_operations":
				atomic.LoadInt64(
					&aiOperationCounter,
				),
		},
	)
}

func apiTestAI(
	w http.ResponseWriter,
	r *http.Request,
) {

	if r.Method != http.MethodPost {

		jsonResponse(
			w,
			405,
			map[string]string{
				"error":
					"method not allowed",
			},
		)

		return
	}

	var req struct {
		Message string `json:"message"`
	}

	if err := json.NewDecoder(
		r.Body,
	).Decode(&req); err != nil {

		jsonResponse(
			w,
			400,
			map[string]string{
				"error":
					"JSON 格式错误",
			},
		)

		return
	}

	if strings.TrimSpace(
		req.Message,
	) == "" {

		req.Message =
			"你好，请简单介绍一下你自己。"
	}

	answer, err :=
		aiBot.RawText(
			req.Message,
		)

	if err != nil {

		jsonResponse(
			w,
			500,
			map[string]interface{}{
				"success": false,
				"error":
					err.Error(),
			},
		)

		return
	}

	compressMemoryIfNeeded()

	jsonResponse(
		w,
		200,
		map[string]interface{}{
			"success": true,
			"answer":
				answer,
		},
	)
}

func apiMemory(
	w http.ResponseWriter,
	r *http.Request,
) {

	if r.Method != http.MethodGet {

		jsonResponse(
			w,
			405,
			map[string]string{
				"error":
					"method not allowed",
			},
		)

		return
	}

	memory, err :=
		readMemoryString()

	if err != nil {

		jsonResponse(
			w,
			500,
			map[string]string{
				"error":
					err.Error(),
			},
		)

		return
	}

	jsonResponse(
		w,
		200,
		map[string]interface{}{
			"size":
				len([]byte(memory)),

			"content":
				memory,
		},
	)
}

func apiCompressMemory(
	w http.ResponseWriter,
	r *http.Request,
) {

	if r.Method != http.MethodPost {

		jsonResponse(
			w,
			405,
			map[string]string{
				"error":
					"method not allowed",
			},
		)

		return
	}

	go compressMemory()

	jsonResponse(
		w,
		200,
		map[string]interface{}{
			"success": true,
			"message":
				"已开始 AI 记忆压缩",
		},
	)
}

// ============================================================
// WEB UI
// ============================================================

func webIndex(
	w http.ResponseWriter,
	r *http.Request,
) {

	w.Header().Set(
		"Content-Type",
		"text/html; charset=utf-8",
	)

	fmt.Fprint(
		w,
		iosHTML,
	)
}

// ============================================================
// FORMAT
// ============================================================

func formatBytes(
	n int64,
) string {

	if n < 1024 {
		return fmt.Sprintf(
			"%d B",
			n,
		)
	}

	units :=
		[]string{
			"KB",
			"MB",
			"GB",
		}

	value :=
		float64(n)

	for _, unit :=
		range units {

		value /= 1024

		if value < 1024 {

			return fmt.Sprintf(
				"%.2f %s",
				value,
				unit,
			)
		}
	}

	return fmt.Sprintf(
		"%.2f TB",
		value,
	)
}

// ============================================================
// WEBSOCKET
// ============================================================

type WSConn struct {
	conn net.Conn

	reader *bufio.Reader

	writeMu sync.Mutex
}

func DialWebSocket(
	rawURL string,
) (*WSConn, error) {

	u, err :=
		url.Parse(rawURL)

	if err != nil {
		return nil, err
	}

	if u.Scheme != "ws" &&
		u.Scheme != "wss" {

		return nil,
			errors.New(
				"WebSocket 只支持 ws/wss",
			)
	}

	host :=
		u.Host

	var conn net.Conn

	dialer :=
		&net.Dialer{
			Timeout:
				20 * time.Second,
		}

	if u.Scheme == "wss" {

		tlsDialer :=
			&tls.Dialer{
				NetDialer: dialer,

				Config:
					&tls.Config{
						ServerName:
							u.Hostname(),

						MinVersion:
							tls.VersionTLS12,
					},
			}

		host = normalizeGatewayAddress(host)

	if tlsDialer != nil {

		conn, err =
			tlsDialer.Dial(
				"tcp",
				host,
			)
	}

} else {

		conn, err =
			dialer.Dial(
				"tcp",
				host,
			)
	}

	if err != nil {
		return nil, err
	}

	keyBytes :=
		make([]byte, 16)

	if _, err := rand.Read(
		keyBytes,
	); err != nil {

		conn.Close()

		return nil, err
	}

	key :=
		base64.StdEncoding.EncodeToString(
			keyBytes,
		)

	path :=
		u.RequestURI()

	if path == "" {
		path = "/"
	}

	var request strings.Builder

	request.WriteString(
		"GET " +
			path +
			" HTTP/1.1\r\n",
	)

	request.WriteString(
		"Host: " +
			u.Host +
			"\r\n",
	)

	request.WriteString(
		"Upgrade: websocket\r\n",
	)

	request.WriteString(
		"Connection: Upgrade\r\n",
	)

	request.WriteString(
		"Sec-WebSocket-Key: " +
			key +
			"\r\n",
	)

	request.WriteString(
		"Sec-WebSocket-Version: 13\r\n",
	)

	request.WriteString(
		"User-Agent: " +
			AppName +
			"/" +
			AppVersion +
			"\r\n",
	)

	request.WriteString(
		"\r\n",
	)

	if _, err := conn.Write(
		[]byte(
			request.String(),
		),
	); err != nil {

		conn.Close()

		return nil, err
	}

	reader :=
		bufio.NewReader(conn)

	statusLine, err :=
		reader.ReadString('\n')

	if err != nil {

		conn.Close()

		return nil, err
	}

	if !strings.Contains(
		statusLine,
		"101",
	) {

		conn.Close()

		return nil,
			fmt.Errorf(
				"WebSocket Upgrade 失败: %s",
				strings.TrimSpace(
					statusLine,
				),
			)
	}

	headers :=
		map[string]string{}

	for {

		line, err :=
			reader.ReadString(
				'\n',
			)

		if err != nil {

			conn.Close()

			return nil, err
		}

		line =
			strings.TrimRight(
				line,
				"\r\n",
			)

		if line == "" {
			break
		}

		parts :=
			strings.SplitN(
				line,
				":",
				2,
			)

		if len(parts) == 2 {

			headers[
				strings.ToLower(
					strings.TrimSpace(
						parts[0],
					),
				),
			] =
				strings.TrimSpace(
					parts[1],
				)
		}
	}

	expected :=
		websocketAccept(key)

	if headers[
		"sec-websocket-accept",
	] != expected {

		conn.Close()

		return nil,
			errors.New(
				"WebSocket Accept 校验失败",
			)
	}

	return &WSConn{
		conn:   conn,
		reader: reader,
	}, nil
}

func websocketAccept(
	key string,
) string {

	const magic =
		"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

	hash :=
		sha1.Sum(
			[]byte(
				key + magic,
			),
		)

	return base64.StdEncoding.EncodeToString(
		hash[:],
	)
}

func (w *WSConn) Close() error {

	if w == nil ||
		w.conn == nil {

		return nil
	}

	return w.conn.Close()
}

func (w *WSConn) WriteMessage(
	payload []byte,
) error {

	w.writeMu.Lock()

	defer w.writeMu.Unlock()

	return w.writeFrame(
		0x1,
		payload,
	)
}

func (w *WSConn) writeFrame(
	opcode byte,
	payload []byte,
) error {

	if len(payload) >
		0x7FFFFFFFFFFFFFFF {

		return errors.New(
			"WebSocket payload too large",
		)
	}

	first :=
		byte(
			0x80 | opcode,
		)

	header :=
		[]byte{
			first,
		}

	length :=
		len(payload)

	switch {

	case length < 126:

		header =
			append(
				header,
				byte(
					0x80|length,
				),
			)

	case length <= 65535:

		header =
			append(
				header,
				0x80|126,
				0,
				0,
			)

		binary.BigEndian.PutUint16(
			header[len(header)-2:],
			uint16(length),
		)

	default:

		header =
			append(
				header,
				0x80|127,
				0, 0, 0, 0,
				0, 0, 0, 0,
			)

		binary.BigEndian.PutUint64(
			header[len(header)-8:],
			uint64(length),
		)
	}

	mask :=
		make([]byte, 4)

	if _, err := rand.Read(
		mask,
	); err != nil {

		return err
	}

	header =
		append(
			header,
			mask...,
		)

	masked :=
		make([]byte, len(payload))

	for i :=
		range payload {

		masked[i] =
			payload[i] ^
				mask[
					i%4,
				]
	}

	if _, err := w.conn.Write(
		header,
	); err != nil {

		return err
	}

	_, err :=
		w.conn.Write(
			masked,
		)

	return err
}

func (w *WSConn) ReadMessage() (
	byte,
	[]byte,
	error,
) {

	var full []byte

	var opcode byte

	for {

		first,
			second,
			err :=
			w.readHeader()

		if err != nil {
			return 0, nil, err
		}

		fin :=
			first&0x80 != 0

		op :=
			first & 0x0f

		masked :=
			second&0x80 != 0

		length :=
			uint64(
				second & 0x7f,
			)

		if length == 126 {

			var b [2]byte

			if _, err :=
				io.ReadFull(
					w.reader,
					b[:],
				); err != nil {

				return 0, nil, err
			}

			length =
				uint64(
					binary.BigEndian.Uint16(
						b[:],
					),
				)
		}

		if length == 127 {

			var b [8]byte

			if _, err :=
				io.ReadFull(
					w.reader,
					b[:],
				); err != nil {

				return 0, nil, err
			}

			length =
				binary.BigEndian.Uint64(
					b[:],
				)

			if length >
				100*1024*1024 {

				return 0,
					nil,
					errors.New(
						"WebSocket 单消息过大",
					)
			}
		}

		var mask [4]byte

		if masked {

			if _, err :=
				io.ReadFull(
					w.reader,
					mask[:],
				); err != nil {

				return 0, nil, err
			}
		}

		data :=
			make([]byte, int(length))

		if _, err :=
			io.ReadFull(
				w.reader,
				data,
			); err != nil {

			return 0, nil, err
		}

		if masked {

			for i :=
				range data {

				data[i] ^=
					mask[
						i%4,
					]
			}
		}

		switch op {

		case 0x9:

			w.writeMu.Lock()

			_ = w.writeFrame(
				0xA,
				data,
			)

			w.writeMu.Unlock()

			continue

		case 0xA:
			continue

		case 0x8:
			return 0, nil, io.EOF
		}

		if opcode == 0 {
			opcode = op
		}

		full =
			append(
				full,
				data...,
			)

		if fin {
			return opcode, full, nil
		}
	}
}

func (w *WSConn) readHeader() (
	byte,
	byte,
	error,
) {

	var header [2]byte

	if _, err :=
		io.ReadFull(
			w.reader,
			header[:],
		); err != nil {

		return 0, 0, err
	}

	return header[0],
		header[1],
		nil
}

// ============================================================
// WEB UI
// ============================================================

const iosHTML = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width,
initial-scale=1,
maximum-scale=1,
user-scalable=no">

<title>QbotAI</title>

<style>

*{
box-sizing:border-box;
-webkit-tap-highlight-color:transparent;
}

body{
margin:0;
font-family:
-apple-system,
BlinkMacSystemFont,
"Helvetica Neue",
Arial,
sans-serif;

background:
linear-gradient(
#cfcfcf,
#eeeeee 40%,
#bcbcbc
);

color:#111;

padding-bottom:30px;
}

.status{
height:22px;

background:
linear-gradient(
#444,
#111
);

color:white;

text-align:center;

font-size:12px;

line-height:22px;

text-shadow:
0 -1px #000;
}

.nav{
height:46px;

display:flex;
align-items:center;
justify-content:center;

color:white;

background:
linear-gradient(
#7197bf,
#365d85 48%,
#274766 52%,
#5f85aa
);

border-top:
1px solid #91aeca;

border-bottom:
1px solid #17283a;

box-shadow:
inset 0 1px rgba(255,255,255,.45),
0 2px 4px rgba(0,0,0,.45);

position:sticky;
top:0;
z-index:100;
}

.nav h1{
font-size:20px;
margin:0;

text-shadow:
0 1px #000;
}

.container{
max-width:900px;
margin:auto;
padding:12px;
}

.card{
background:
linear-gradient(
#fff,
#e8e8e8
);

border:
1px solid #aaa;

border-radius:10px;

overflow:hidden;

margin-bottom:12px;

box-shadow:
0 1px 3px rgba(0,0,0,.45),
inset 0 1px white;
}

.title{
padding:8px 12px;

font-size:13px;

font-weight:bold;

color:#333;

background:
linear-gradient(
#ededed,
#c8c8c8
);

border-bottom:
1px solid #aaa;

text-shadow:
0 1px white;
}

.row{
padding:10px 12px;

border-bottom:
1px solid #d0d0d0;
}

.row:last-child{
border-bottom:0;
}

label{
display:block;

font-size:12px;

font-weight:bold;

color:#555;

margin-bottom:5px;
}

input,
textarea{
width:100%;

border:
1px solid #999;

border-radius:7px;

background:#fff;

padding:9px;

font-size:14px;

box-shadow:
inset 0 1px 3px rgba(0,0,0,.25);

outline:none;
}

textarea{
min-height:120px;
resize:vertical;
}

input:focus,
textarea:focus{
border-color:#3e72a5;

box-shadow:
0 0 4px rgba(40,100,160,.5),
inset 0 1px 3px rgba(0,0,0,.2);
}

.switch{
display:flex;
align-items:center;
justify-content:space-between;
}

.switch label{
margin:0;
}

.switch input{
width:52px;
height:30px;

appearance:none;

border-radius:16px;

padding:0;

background:#aaa;

position:relative;
}

.switch input:before{
content:"";

position:absolute;

width:26px;
height:26px;

left:1px;
top:1px;

border-radius:50%;

background:
linear-gradient(
#fff,
#ccc
);

box-shadow:
0 1px 2px #555;

transition:.15s;
}

.switch input:checked{
background:
linear-gradient(
#65c84c,
#36a72a
);
}

.switch input:checked:before{
left:24px;
}

button{
border:
1px solid #777;

border-radius:8px;

padding:9px 15px;

font-weight:bold;

background:
linear-gradient(
#fff,
#d0d0d0
);

box-shadow:
inset 0 1px white,
0 1px 2px rgba(0,0,0,.35);

margin:3px;

cursor:pointer;
}

button.blue{
color:white;

text-shadow:
0 -1px #234;

border-color:#284f76;

background:
linear-gradient(
#72a0d2,
#356896 50%,
#2e5c89
);
}

button.red{
color:white;

background:
linear-gradient(
#e96969,
#a52626
);
}

.statusBox{
padding:12px;
text-align:center;
font-size:13px;
}

.dot{
display:inline-block;

width:10px;
height:10px;

border-radius:50%;

background:#888;

margin-right:4px;

box-shadow:
inset 0 1px white,
0 1px 2px #555;
}

.dot.on{
background:#36bd31;
}

.toolbar{
padding:5px;

display:flex;
flex-wrap:wrap;
}

.small{
font-size:11px;
color:#777;
margin-top:4px;
}

#toast{
display:none;

position:fixed;

left:50%;
bottom:25px;

transform:
translateX(-50%);

background:
linear-gradient(
#333,
#111
);

color:white;

padding:10px 16px;

border-radius:9px;

box-shadow:
0 2px 8px #333;

z-index:999;
}

.footer{
text-align:center;

font-size:11px;

color:#555;

padding:15px;

text-shadow:
0 1px white;
}

</style>
</head>

<body>

<div class="status">
QbotAI
</div>

<div class="nav">
<h1>QbotAI</h1>
</div>

<div class="container">

<div class="card">
<div class="title">运行状态</div>

<div class="statusBox">

<span id="qqDot" class="dot"></span>
QQ：
<span id="qqStatus">检测中</span>

&nbsp;&nbsp;

<span id="aiDot" class="dot"></span>
AI：
<span id="aiStatus">检测中</span>

<br><br>

Memory：
<span id="memorySize">-</span>

</div>
</div>

<div class="card">

<div class="title">
QQ Bot
</div>

<div class="row">
<label>App ID / Bot ID</label>
<input id="qqAppID">
</div>

<div class="row">
<label>Client Secret</label>
<input id="qqSecret" type="password">
</div>

<div class="row switch">
<label>启用 QQ Bot</label>
<input id="qqEnabled"
type="checkbox">
</div>

</div>

<div class="card">

<div class="title">
AI
</div>

<div class="row switch">
<label>启用 AI</label>
<input id="aiEnabled"
type="checkbox">
</div>

<div class="row">
<label>API Base URL</label>
<input id="aiBaseURL"
placeholder="https://api.openai.com/v1">

<div class="small">
最终请求 /chat/completions
</div>
</div>

<div class="row">
<label>API Key</label>
<input id="aiKey"
type="password">
</div>

<div class="row">
<label>Model</label>
<input id="aiModel">
</div>

<div class="row">
<label>System Prompt</label>
<textarea id="aiPrompt"></textarea>
</div>

<div class="row">
<label>Temperature</label>
<input id="temperature"
type="number"
min="0"
max="2"
step="0.1">
</div>

<div class="row">
<label>Max Tokens</label>
<input id="maxTokens"
type="number">
</div>

<div class="row switch">
<label>AI 使用长期记忆</label>
<input id="memoryEnabled"
type="checkbox">
</div>

</div>

<div class="card">

<div class="title">
群聊 AI
</div>

<div class="row switch">
<label>普通群聊 AI 观察</label>
<input id="groupEnabled"
type="checkbox">
</div>

<div class="row">
<label>上下文消息数量</label>
<input id="contextMessages"
type="number">
</div>

<div class="row">
<label>AI 判断延迟（秒）</label>
<input id="decisionDelay"
type="number">
</div>

<div class="row">
<label>回复冷却（秒）</label>
<input id="replyCooldown"
type="number">
</div>

<div class="row">
<label>每小时最大回复</label>
<input id="maxReplies"
type="number">
</div>

<div class="row">
<label>主动发言程度 0-100</label>
<input id="proactive"
type="number"
min="0"
max="100">
</div>

</div>

<div class="card">

<div class="title">
长期记忆
</div>

<div class="row switch">
<label>启用 Markdown Memory</label>
<input id="memoryEnabled2"
type="checkbox">
</div>

<div class="row">
<label>每多少次操作压缩</label>
<input id="compressionEvery"
type="number">
</div>

<div class="row">
<label>压缩后目标比例</label>
<input id="compressionRatio"
type="number"
min="0.1"
max="0.99"
step="0.01">
<div class="small">
0.75 = 压缩到原来的75%，即减少约25%
</div>
</div>

<div class="row">
<label>最大大小（GB）</label>
<input id="maxGB"
type="number"
step="0.1">
</div>

<div class="row">

<button
class="blue"
onclick="compressMemory()">
立即 AI 压缩记忆
</button>

</div>

</div>

<div class="card">

<div class="title">
AI 测试
</div>

<div class="row">
<label>输入</label>

<textarea
id="testMessage">你好，请介绍一下QbotAI。</textarea>
</div>

<div class="row">
<label>AI 返回</label>

<textarea
id="testResult"
readonly></textarea>
</div>

<div class="row">

<button
class="blue"
onclick="testAI()">
测试 AI
</button>

</div>

</div>

<div class="card">

<div class="title">
操作
</div>

<div class="toolbar">

<button
class="blue"
onclick="saveConfig()">
保存配置
</button>

<button
onclick="loadConfig()">
重新读取
</button>

</div>

</div>

<div class="footer">
QbotAI 2.0.0<br>
Go Cross Platform
</div>

</div>

<div id="toast"></div>

<script>

function $(id){
return document.getElementById(id);
}

function toast(text){

let t=$("toast");

t.innerText=text;

t.style.display="block";

clearTimeout(window.__toast);

window.__toast=setTimeout(
()=>{
t.style.display="none";
},
1800
);
}

async function loadConfig(){

try{

let r=
await fetch("/api/config");

let c=
await r.json();

$("qqAppID").value=
c.qq.app_id || "";

$("qqSecret").value=
c.qq.client_secret || "";

$("qqEnabled").checked=
!!c.qq.enabled;

$("aiEnabled").checked=
!!c.ai.enabled;

$("aiBaseURL").value=
c.ai.base_url || "";

$("aiKey").value=
c.ai.api_key || "";

$("aiModel").value=
c.ai.model || "";

$("aiPrompt").value=
c.ai.system_prompt || "";

$("temperature").value=
c.ai.temperature ?? .7;

$("maxTokens").value=
c.ai.max_tokens ?? 1200;

$("memoryEnabled").checked=
!!c.ai.memory_enabled;

$("memoryEnabled2").checked=
!!c.memory.enabled;

$("groupEnabled").checked=
!!c.group_ai.enabled;

$("contextMessages").value=
c.group_ai.context_messages ?? 20;

$("decisionDelay").value=
c.group_ai.decision_delay_seconds ?? 3;

$("replyCooldown").value=
c.group_ai.reply_cooldown_seconds ?? 30;

$("maxReplies").value=
c.group_ai.max_replies_per_hour ?? 30;

$("proactive").value=
c.group_ai.proactive_level ?? 40;

$("compressionEvery").value=
c.memory.compression_every ?? 15;

$("compressionRatio").value=
c.memory.compression_ratio ?? .75;

$("maxGB").value=
(c.memory.max_bytes || 2147483648) /
1024 /
1024 /
1024;

toast("读取成功");

}catch(e){

toast("读取失败");

}

}

async function saveConfig(){

let maxGB=
Number(
$("maxGB").value
);

if(!maxGB || maxGB<=0){
maxGB=2;
}

let data={

qq:{
app_id:
$("qqAppID").value,

client_secret:
$("qqSecret").value,

enabled:
$("qqEnabled").checked
},

ai:{
enabled:
$("aiEnabled").checked,

base_url:
$("aiBaseURL").value,

api_key:
$("aiKey").value,

model:
$("aiModel").value,

system_prompt:
$("aiPrompt").value,

temperature:
Number(
$("temperature").value
),

max_tokens:
Number(
$("maxTokens").value
),

memory_enabled:
$("memoryEnabled").checked
},

group_ai:{

enabled:
$("groupEnabled").checked,

context_messages:
Number(
$("contextMessages").value
),

decision_delay_seconds:
Number(
$("decisionDelay").value
),

reply_cooldown_seconds:
Number(
$("replyCooldown").value
),

max_replies_per_hour:
Number(
$("maxReplies").value
),

proactive_level:
Number(
$("proactive").value
),

only_reply_when_useful:
true
},

memory:{

enabled:
$("memoryEnabled2").checked,

compression_every:
Number(
$("compressionEvery").value
),

compression_ratio:
Number(
$("compressionRatio").value
),

max_bytes:
Math.floor(
maxGB *
1024 *
1024 *
1024
),

delete_oldest_first:
true
},

web:{
host:"127.0.0.1",
port:8080
},

bot:{
name:"QbotAI",
prefix:"/"
}

};

try{

let r=
await fetch(
"/api/save",
{
method:"POST",
headers:{
"Content-Type":
"application/json"
},
body:
JSON.stringify(data)
}
);

let result=
await r.json();

if(result.success){
toast("保存成功");
}else{
toast(
result.error ||
"保存失败"
);
}

}catch(e){

toast("保存失败");

}

}

async function testAI(){

$("testResult").value=
"正在请求 AI...";

try{

let r=
await fetch(
"/api/test-ai",
{
method:"POST",
headers:{
"Content-Type":
"application/json"
},
body:
JSON.stringify({
message:
$("testMessage").value
})
}
);

let result=
await r.json();

if(result.success){

$("testResult").value=
result.answer || "";

}else{

$("testResult").value=
result.error ||
"AI 请求失败";

}

}catch(e){

$("testResult").value=
String(e);

}

}

async function compressMemory(){

try{

let r=
await fetch(
"/api/compress-memory",
{
method:"POST"
}
);

let result=
await r.json();

toast(
result.message ||
"已开始压缩"
);

}catch(e){

toast("压缩请求失败");

}

}

async function updateStatus(){

try{

let r=
await fetch(
"/api/status"
);

let s=
await r.json();

$("qqDot").className=
s.qq_online ?
"dot on":
"dot";

$("aiDot").className=
s.ai_enabled ?
"dot on":
"dot";

$("qqStatus").innerText=
s.qq_online ?
"在线":
"离线";

$("aiStatus").innerText=
s.ai_enabled ?
"启用":
"关闭";

$("memorySize").innerText=
formatBytes(
s.memory_size
);

}catch(e){

$("qqStatus").innerText=
"错误";

$("aiStatus").innerText=
"错误";

}

}

function formatBytes(n){

if(n<1024)
return n+" B";

if(n<1024*1024)
return (n/1024).toFixed(2)+" KB";

if(n<1024*1024*1024)
return (n/1024/1024).toFixed(2)+" MB";

return (
n/1024/1024/1024
).toFixed(2)+" GB";

}

loadConfig();

updateStatus();

setInterval(
updateStatus,
3000
);

</script>

</body>
</html>`
