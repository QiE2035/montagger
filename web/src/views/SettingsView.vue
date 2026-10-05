<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import {
  NButton,
  NCard,
  NInput,
  NInputNumber,
  NModal,
  NSelect,
  NSpin,
  NSwitch,
  NTag,
  useMessage,
} from "naive-ui";
import {
  Check,
  Download,
  HardDriveDownload,
  KeyRound,
  MemoryStick,
  Plus,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-vue-next";
import { api, type ModelStatus, type ModelsResponse, type Settings, type Token } from "../api";

const message = useMessage();
const loading = ref(true);
const models = ref<ModelsResponse | null>(null);
const settings = ref<Settings | null>(null);
const tokens = ref<Token[]>([]);
const installing = reactive<Record<string, boolean>>({});
const newPassword = ref("");
const newTokenName = ref("");
const newHfToken = ref("");
const hfEndpoint = ref("");
const serverBind = ref("");
const serverBaseUrl = ref("");
const modelPath = ref("");
const monbooruUrl = ref("");
const monbooruStatus = reactive({
  paired: false,
  waiting: false,
  push_tags: false,
  push_images: false,
  gallery: "",
});
const galleryOptions = ref<{ label: string; value: string }[]>([]);
const health = ref<{ rss_mb: number; models_loaded: { name: string; idle_s: number }[] } | null>(null);
const releasing = ref(false);
let statusTimer: ReturnType<typeof setInterval> | null = null;

// -- threshold drafts (monbooru-style dialog) ---------------------------------

interface CatDraft {
  thr: number | null; // null = use default
  topk: number | null; // null = use default top-k
  off: boolean; // disabled: emit nothing for this category
}

interface ThrDraft {
  global: number;
  cats: Record<string, CatDraft>;
}

const drafts = reactive<Record<string, ThrDraft>>({});
const dialogModel = ref<ModelStatus | null>(null);

function draftFor(m: ModelStatus): ThrDraft {
  const existing = drafts[m.name];
  if (existing) return existing;
  const cats: Record<string, CatDraft> = {};
  const names = m.emitted_categories.length
    ? m.emitted_categories
    : Object.keys(m.effective.categories);
  for (const cat of names) {
    cats[cat] = {
      thr: m.effective.categories[cat] ?? null,
      topk: m.effective.top_k[cat] ?? null,
      off: m.effective.disabled.includes(cat),
    };
  }
  const draft: ThrDraft = { global: m.effective.global, cats };
  drafts[m.name] = draft;
  return draft;
}

const dialogDraft = computed<ThrDraft | null>(() =>
  dialogModel.value ? draftFor(dialogModel.value) : null,
);

function openThresholds(m: ModelStatus) {
  dialogModel.value = m;
}

async function saveThresholds() {
  const m = dialogModel.value;
  const draft = m ? drafts[m.name] : null;
  if (!m || !draft) return;
  const categories: Record<string, number> = {};
  const top_k: Record<string, number> = {};
  const disabled: string[] = [];
  for (const [cat, c] of Object.entries(draft.cats)) {
    if (c.off) {
      disabled.push(cat);
      continue;
    }
    if (c.thr !== null) categories[cat] = c.thr;
    if (c.topk !== null) top_k[cat] = c.topk;
  }
  await save({ thresholds: { [m.name]: { global: draft.global, categories, top_k, disabled } } });
  message.success(`${m.name} 阈值已保存`);
  dialogModel.value = null;
  await load();
}

// -- load / refresh ------------------------------------------------------------

async function load() {
  loading.value = true;
  try {
    [models.value, settings.value, tokens.value] = await Promise.all([
      api.models(),
      api.settings(),
      api.tokens(),
    ]);
    for (const m of models.value.models) draftFor(m); // seed drafts from effective
    hfEndpoint.value = settings.value?.hf.endpoint ?? "";
    serverBind.value = settings.value?.server.bind_address ?? "";
    serverBaseUrl.value = settings.value?.server.base_url ?? "";
    modelPath.value = settings.value?.models.path ?? "";
    try {
      health.value = await api.get("/health");
    } catch {
      health.value = null;
    }
    await refreshMonbooru();
  } catch (err) {
    message.error(String((err as Error).message));
  } finally {
    loading.value = false;
  }
}

async function refreshMonbooru() {
  try {
    const status = await api.monbooruStatus();
    Object.assign(monbooruStatus, status);
    monbooruUrl.value = monbooruUrl.value || status.api_url;
    if (status.paired) {
      const { galleries } = await api.monbooruGalleries();
      galleryOptions.value = galleries.map((g: { name: string }) => ({ label: g.name, value: g.name }));
    }
  } catch {
    /* pairing surfaces come up when the backend does */
  }
}

async function pairMonbooru() {
  try {
    await api.monbooruPair(monbooruUrl.value.trim());
    message.success("配对请求已发送 · 到 monbooru 的 Settings → Plugins 批准");
    pollStatus();
  } catch (err) {
    message.error(String((err as Error).message));
  }
}

async function unpairMonbooru() {
  await api.monbooruUnpair();
  message.success("已取消配对");
  refreshMonbooru();
}

function pollStatus() {
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = setInterval(async () => {
    await refreshMonbooru();
    if (monbooruStatus.paired) {
      if (statusTimer) clearInterval(statusTimer);
      statusTimer = null;
      message.success("monbooru 配对成功");
    }
  }, 3000);
}

onBeforeUnmount(() => {
  if (statusTimer) clearInterval(statusTimer);
});
onMounted(load);

// -- model actions --------------------------------------------------------------

async function install(name: string) {
  installing[name] = true;
  try {
    await api.installModel(name);
    message.success(`${name} 安装完成`);
    await load();
  } catch (err) {
    message.error(String((err as Error).message));
  } finally {
    installing[name] = false;
  }
}

async function removeModel(name: string) {
  try {
    await api.removeModel(name);
    message.success(`${name} 已删除`);
    await load();
  } catch (err) {
    message.error(String((err as Error).message));
  }
}

// -- settings plumbing ------------------------------------------------------------

async function save(patch: object) {
  try {
    await api.saveSettings(patch);
    settings.value = await api.settings();
    // monbooru controls bind to monbooruStatus (the polled view of the
    // world); refresh it now or switches sit frozen until the next poll.
    if ("monbooru" in patch) await refreshMonbooru();
  } catch (err) {
    message.error(String((err as Error).message));
    throw err;
  }
}

const loadedNames = computed(
  () => new Set((health.value?.models_loaded ?? []).map((x) => x.name)),
);

async function releaseMemory(names?: string[]) {
  releasing.value = true;
  try {
    const r = await api.releaseMemory(names);
    health.value = { rss_mb: r.rss_mb, models_loaded: [] };
    message.success(
      names?.length
        ? `${names[0]} 已卸载 · 常驻 ${r.rss_mb.toFixed(0)} MB`
        : `已释放：卸载 ${r.unloaded.length} 个模型 · 常驻 ${r.rss_mb.toFixed(0)} MB`,
    );
    await load();
  } catch (err) {
    message.error(String((err as Error).message));
  } finally {
    releasing.value = false;
  }
}

async function purgeHistory() {
  try {
    const r = await api.purgeHistory();
    message.success(`已清理 ${r.deleted} 条过期历史`);
  } catch (err) {
    message.error(String((err as Error).message));
  }
}

async function applyPassword() {
  const value = newPassword.value;
  try {
    await save({ auth: { password: value } });
    newPassword.value = "";
    message.success(value === "" ? "已关闭密码" : "密码已更新");
  } catch {
    /* message shown by save() */
  }
}

async function applyHfEndpoint() {
  try {
    await save({ hf: { endpoint: hfEndpoint.value.trim() } });
    message.success("HF 端点已更新");
  } catch {
    /* message shown by save() */
  }
}

async function applyHfToken() {
  const value = newHfToken.value;
  try {
    await save({ hf: { token: value } });
    newHfToken.value = "";
    message.success(value === "" ? "已清除 HF Token" : "HF Token 已更新");
  } catch {
    /* message shown by save() */
  }
}

async function applyServer() {
  try {
    await save({
      server: { bind_address: serverBind.value.trim(), base_url: serverBaseUrl.value.trim() },
    });
    message.success("服务器设置已保存 · 监听地址重启后生效");
  } catch {
    /* message shown by save() */
  }
}

async function applyModelPath() {
  try {
    await save({ models: { path: modelPath.value.trim() } });
    message.success("模型目录已保存 · 引擎已重新加载");
  } catch {
    /* message shown by save() */
  }
}

async function createToken() {
  try {
    const { token } = await api.createToken(newTokenName.value || "unnamed");
    newTokenName.value = "";
    await loadTokens();
    message.success(`已创建：${token.slice(0, 12)}…（完整值见列表）`);
  } catch (err) {
    message.error(String((err as Error).message));
  }
}

async function loadTokens() {
  tokens.value = await api.tokens();
}

async function deleteToken(id: number) {
  await api.deleteToken(id);
  await loadTokens();
}

// -- derived options --------------------------------------------------------------

const defaultModelOptions = computed(() => {
  const list = (models.value?.models ?? []).map((m) => ({
    label: m.name + (m.available ? "" : `（${m.reason}）`),
    value: m.name,
  }));
  for (const name of settings.value?.models.default_models ?? []) {
    if (!list.some((o) => o.value === name)) list.push({ label: name, value: name });
  }
  return list;
});

const defaultModelsValue = computed(() => {
  const configured = settings.value?.models.default_models ?? [];
  if (configured.length) return configured;
  return settings.value?.models.default ? [settings.value.models.default] : [];
});

function setDefaultModels(names: string[]) {
  save({ models: { default_models: names } }).catch(() => undefined);
}

function fmtSize(bytes: number): string {
  if (!bytes) return "—";
  return `${(bytes / 1e6).toFixed(0)} MB`;
}

const categoryZh: Record<string, string> = {
  general: "通用",
  character: "角色",
  copyright: "作品",
  artist: "画师",
  meta: "元信息",
  rating: "分级",
  medium: "媒介",
  person: "人物",
  species: "物种",
  year: "年份",
};

const categoryOptions = Object.entries(categoryZh).map(([value, label]) => ({ label, value }));
</script>

<template>
  <NSpin :show="loading">
    <div class="flex flex-col gap-5">
      <h1 class="px-1 text-lg font-semibold">设置</h1>

      <!-- Models -->
      <section class="flex flex-col gap-3">
        <h2 class="px-1 text-sm font-medium text-[var(--mt-text-dim)]">模型</h2>
        <NCard
          v-for="m in models?.models ?? []"
          :key="m.name"
          size="small"
          class="!rounded-2xl"
        >
          <div class="flex flex-wrap items-center gap-2">
            <span class="font-medium">{{ m.name }}</span>
            <NTag v-if="defaultModelsValue.includes(m.name)" size="small" type="primary" round>
              默认{{ defaultModelsValue.length > 1 ? `之一` : "" }}
            </NTag>
            <NTag size="small" round :type="m.available ? 'success' : m.reason === 'not installed' ? 'default' : 'warning'">
              {{ m.available ? "可用" : m.reason }}
            </NTag>
            <NTag v-if="loadedNames.has(m.name)" size="small" round type="info">已加载</NTag>
            <NTag v-if="m.gated" size="small" round type="warning">需 HF Token</NTag>
            <span class="text-xs text-[var(--mt-text-dim)]">{{ fmtSize(m.size_bytes) }}</span>
            <div class="ml-auto flex gap-1.5">
              <NButton
                v-if="!m.available"
                size="tiny"
                type="primary"
                :loading="installing[m.name]"
                @click="install(m.name)"
              >
                <template #icon><HardDriveDownload :size="13" /></template>
                安装
              </NButton>
              <NButton v-if="m.available" size="tiny" @click="openThresholds(m)">
                <template #icon><SlidersHorizontal :size="13" /></template>
                阈值
              </NButton>
              <NButton
                v-if="loadedNames.has(m.name)"
                size="tiny"
                quaternary
                :disabled="releasing"
                @click="releaseMemory([m.name])"
              >
                <template #icon><MemoryStick :size="13" /></template>
                卸载
              </NButton>
              <NButton v-if="m.discovered" size="tiny" quaternary type="error" @click="removeModel(m.name)">
                <template #icon><Trash2 :size="13" /></template>
              </NButton>
            </div>
          </div>
          <p v-if="m.description" class="mt-1 text-xs text-[var(--mt-text-dim)]">{{ m.description }}</p>
        </NCard>
      </section>

      <!-- Inference -->
      <section v-if="settings" class="flex flex-col gap-3">
        <h2 class="px-1 text-sm font-medium text-[var(--mt-text-dim)]">推理</h2>
        <NCard size="small" class="!rounded-2xl">
          <div class="flex flex-wrap items-end gap-4">
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              默认打标模型（多选，上传时每张图各打一遍）
              <NSelect
                :value="defaultModelsValue"
                :options="defaultModelOptions"
                multiple
                clearable
                style="width: 280px"
                placeholder="选择一个或多个模型"
                @update:value="setDefaultModels"
              />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              执行提供者（显式选择，保存后重启生效）
              <NSelect
                :value="settings.models.execution_provider"
                :options="(settings.models.available_providers.map((p) => ({ label: p.replace('ExecutionProvider', '').toLowerCase(), value: p.replace('ExecutionProvider', '').toLowerCase() })))"
                style="width: 200px"
                @update:value="(v: string) => save({ models: { execution_provider: v } })"
              />
            </label>
            <span class="mb-2 text-xs" :class="settings.models.running_provider ? 'text-green-400' : 'text-red-400'">
              运行中：{{ settings.models.running_provider || "未加载" }}
            </span>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              设备 ID
              <NInputNumber
                :value="settings.models.device_id"
                size="small"
                :min="0"
                style="width: 110px"
                @update:value="(v: number | null) => save({ models: { device_id: v ?? 0 } })"
              />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              推理线程数（0 = 默认）
              <NInputNumber
                :value="settings.models.intra_op_threads"
                size="small"
                :min="0"
                :max="64"
                style="width: 110px"
                @update:value="(v: number | null) => save({ models: { intra_op_threads: v ?? 0 } })"
              />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              上传上限 (MB)
              <NInputNumber
                :value="settings.models.max_upload_mb"
                size="small"
                :min="1"
                :max="1024"
                style="width: 110px"
                @update:value="(v: number | null) => save({ models: { max_upload_mb: v ?? 100 } })"
              />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              空闲自动卸载（分钟，0 = 不卸载）
              <NInputNumber
                :value="settings.models.idle_unload_min"
                size="small"
                :min="0"
                :max="1440"
                style="width: 140px"
                @update:value="(v: number | null) => save({ models: { idle_unload_min: v ?? 0 } })"
              />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              同时驻留模型数（0 = 不限）
              <NInputNumber
                :value="settings.models.max_loaded"
                size="small"
                :min="0"
                :max="64"
                style="width: 130px"
                @update:value="(v: number | null) => save({ models: { max_loaded: v ?? 1 } })"
              />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              全局禁用类别（不输出）
              <NSelect
                :value="settings.models.disabled_categories"
                :options="categoryOptions"
                multiple
                clearable
                style="width: 260px"
                placeholder="无"
                @update:value="(v: string[]) => save({ models: { disabled_categories: v } })"
              />
            </label>
            <label class="mb-1.5 flex items-center gap-2 text-xs text-[var(--mt-text-dim)]">
              <NSwitch
                :value="settings.models.isolated"
                size="small"
                @update:value="(v: boolean) => save({ models: { isolated: v } })"
              />
              独立推理进程
            </label>
            <NButton size="small" :loading="releasing" @click="releaseMemory()">
              <template #icon><MemoryStick :size="14" /></template>
              立即释放内存
            </NButton>
          </div>
          <p class="mt-2 text-xs text-[var(--mt-text-dim)]">
            模型目录：{{ settings.models.model_root }}
            <template v-if="health"> · 常驻内存 {{ health.rss_mb.toFixed(0) }} MB · 已加载 {{ health.models_loaded.length }} 个模型</template>
          </p>
          <div class="mt-2 flex flex-wrap items-end gap-3">
            <label class="flex flex-1 flex-col gap-1 text-xs text-[var(--mt-text-dim)]" style="min-width: 240px">
              自定义模型目录（空 = 随仓库目录，保存后重新加载引擎）
              <div class="flex gap-2">
                <NInput v-model:value="modelPath" size="small" placeholder="空 = <仓库>/models" />
                <NButton size="small" @click="applyModelPath">
                  <template #icon><Check :size="14" /></template>
                  保存
                </NButton>
              </div>
            </label>
          </div>
        </NCard>
      </section>

      <!-- Queue -->
      <section v-if="settings" class="flex flex-col gap-3">
        <h2 class="px-1 text-sm font-medium text-[var(--mt-text-dim)]">队列</h2>
        <NCard size="small" class="!rounded-2xl">
          <div class="flex flex-wrap items-end gap-4">
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              等待上限（0 = 无上限，重启后生效）
              <NInputNumber
                :value="settings.queue.max_pending"
                size="small"
                :min="0"
                :max="1024"
                style="width: 130px"
                @update:value="(v: number | null) => save({ queue: { max_pending: v ?? 0 } })"
              />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              历史保留（天）
              <NInputNumber
                :value="settings.queue.history_days"
                size="small"
                :min="1"
                :max="365"
                style="width: 120px"
                @update:value="(v: number | null) => save({ queue: { history_days: v ?? 7 } })"
              />
            </label>
            <NButton size="small" @click="purgeHistory">
              <template #icon><Trash2 :size="14" /></template>
              立即清理过期历史
            </NButton>
          </div>
        </NCard>
      </section>

      <!-- Access -->
      <section class="flex flex-col gap-3">
        <h2 class="px-1 text-sm font-medium text-[var(--mt-text-dim)]">访问控制</h2>
        <NCard size="small" class="!rounded-2xl">
          <div v-if="settings && !settings.auth.has_password" class="mb-3 rounded-xl bg-amber-500/10 px-3 py-2 text-xs text-amber-400">
            当前未设置界面密码（开放 LAN 模式）· 仅建议在可信局域网使用
          </div>
          <div class="flex flex-wrap items-end gap-3">
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              {{ settings?.auth.has_password ? "更换密码（留空关闭）" : "设置界面密码" }}
              <NInput v-model:value="newPassword" type="password" show-password-on="click" size="small" placeholder="••••••" style="width: 220px" />
            </label>
            <NButton size="small" :disabled="!newPassword && !settings?.auth.has_password" @click="applyPassword">
              <template #icon><KeyRound :size="14" /></template>
              保存
            </NButton>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              会话有效期（天）
              <NInputNumber
                :value="settings?.auth.session_days ?? 30"
                size="small"
                :min="1"
                :max="365"
                style="width: 110px"
                @update:value="(v: number | null) => save({ auth: { session_days: v ?? 30 } })"
              />
            </label>
          </div>

          <div class="mt-4 flex flex-col gap-2">
            <div class="flex items-center gap-2 text-xs text-[var(--mt-text-dim)]">API Token（手机快捷指令 / 程序推送用）</div>
            <div v-for="t in tokens" :key="t.id" class="flex items-center gap-2 rounded-xl bg-[var(--mt-bg-soft)] px-3 py-2">
              <span class="text-xs">{{ t.name }}</span>
              <code class="truncate text-xs text-[var(--mt-text-dim)]">{{ t.token }}</code>
              <NButton size="tiny" quaternary type="error" class="ml-auto" @click="deleteToken(t.id)">
                <template #icon><Trash2 :size="13" /></template>
              </NButton>
            </div>
            <div class="flex items-center gap-2">
              <NInput v-model:value="newTokenName" size="small" placeholder="名称，如 phone" style="width: 200px" @keyup.enter="createToken" />
              <NButton size="small" @click="createToken">
                <template #icon><Plus :size="14" /></template>
                生成
              </NButton>
            </div>
          </div>
        </NCard>
      </section>

      <!-- Server -->
      <section v-if="settings" class="flex flex-col gap-3">
        <h2 class="px-1 text-sm font-medium text-[var(--mt-text-dim)]">服务器</h2>
        <NCard size="small" class="!rounded-2xl">
          <div class="flex flex-wrap items-end gap-3">
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              监听地址（重启后生效）
              <NInput v-model:value="serverBind" size="small" placeholder="0.0.0.0:8457" style="width: 200px" />
            </label>
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              对外地址（monbooru 配对回调用）
              <NInput v-model:value="serverBaseUrl" size="small" placeholder="http://192.168.1.5:8457" style="width: 240px" />
            </label>
            <NButton size="small" @click="applyServer">
              <template #icon><Check :size="14" /></template>
              保存
            </NButton>
            <label class="mb-1.5 flex items-center gap-2 text-xs text-[var(--mt-text-dim)]">
              <NSwitch
                :value="settings?.log.debug ?? false"
                size="small"
                @update:value="(v: boolean) => save({ log: { debug: v } })"
              />
              调试日志（保存即生效）
            </label>
          </div>
        </NCard>
      </section>

      <!-- HuggingFace -->
      <section v-if="settings" class="flex flex-col gap-3">
        <h2 class="px-1 text-sm font-medium text-[var(--mt-text-dim)]">HuggingFace</h2>
        <NCard size="small" class="!rounded-2xl">
          <div class="flex flex-wrap items-end gap-3">
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              镜像端点（下载模型时生效）
              <div class="flex gap-2">
                <NInput v-model:value="hfEndpoint" size="small" placeholder="https://huggingface.co" style="width: 240px" />
                <NButton size="small" @click="applyHfEndpoint">
                  <template #icon><Check :size="14" /></template>
                  保存
                </NButton>
              </div>
            </label>
          </div>
          <div class="mt-3 flex flex-wrap items-end gap-3">
            <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
              {{ settings?.hf.has_token ? "访问令牌（已设置 · 留空保存即清除）" : "访问令牌（gated 模型下载需要）" }}
              <div class="flex gap-2">
                <NInput
                  v-model:value="newHfToken"
                  type="password"
                  show-password-on="click"
                  size="small"
                  placeholder="hf_..."
                  style="width: 240px"
                />
                <NButton size="small" :disabled="!newHfToken && !settings?.hf.has_token" @click="applyHfToken">
                  <template #icon><KeyRound :size="14" /></template>
                  保存
                </NButton>
              </div>
            </label>
          </div>
          <p class="mt-2 text-xs text-[var(--mt-text-dim)]">
            环境变量 HF_TOKEN / HUGGING_FACE_HUB_TOKEN 优先于此处保存的令牌 · HF_ENDPOINT 优先于端点设置
          </p>
        </NCard>
      </section>

      <!-- monbooru -->
      <section v-if="settings" class="flex flex-col gap-3">
        <h2 class="px-1 text-sm font-medium text-[var(--mt-text-dim)]">monbooru</h2>
        <NCard size="small" class="!rounded-2xl">
          <div class="flex flex-wrap items-end gap-3">
            <label class="flex flex-1 flex-col gap-1 text-xs text-[var(--mt-text-dim)]" style="min-width: 240px">
              monbooru API 地址
              <div class="flex gap-2">
                <NInput v-model:value="monbooruUrl" size="small" placeholder="http://192.168.1.10:8455" />
                <NButton size="small" type="primary" :disabled="!monbooruUrl" @click="pairMonbooru">
                  <template #icon><Download :size="14" /></template>
                  连接
                </NButton>
              </div>
            </label>
            <NTag size="small" round :type="monbooruStatus.paired ? 'success' : monbooruStatus.waiting ? 'warning' : 'default'">
              {{ monbooruStatus.paired ? "已配对" : monbooruStatus.waiting ? "等待批准…" : "未配对" }}
            </NTag>
            <NButton v-if="monbooruStatus.paired" size="small" quaternary type="error" @click="unpairMonbooru">
              取消配对
            </NButton>
          </div>
          <p v-if="monbooruStatus.waiting" class="mt-2 text-xs text-amber-400">
            已发送配对请求 · 到 monbooru 的 Settings → Plugins 批准
          </p>
          <template v-if="monbooruStatus.paired">
            <div class="mt-3 flex flex-wrap items-end gap-4">
              <label class="flex flex-col gap-1 text-xs text-[var(--mt-text-dim)]">
                打标结果写入画廊
                <NSelect
                  :value="monbooruStatus.gallery"
                  clearable
                  :options="galleryOptions"
                  placeholder="不推送"
                  style="width: 220px"
                  @update:value="(v: string | null) => save({ monbooru: { gallery: v ?? '' } })"
                />
              </label>
              <label class="mb-1.5 flex items-center gap-2 text-xs text-[var(--mt-text-dim)]">
                <NSwitch
                  :value="monbooruStatus.push_tags"
                  size="small"
                  @update:value="(v: boolean) => save({ monbooru: { push_tags: v } })"
                />
                推送标签
              </label>
              <label class="mb-1.5 flex items-center gap-2 text-xs text-[var(--mt-text-dim)]">
                <NSwitch
                  :value="monbooruStatus.push_images"
                  size="small"
                  @update:value="(v: boolean) => save({ monbooru: { push_images: v } })"
                />
                推送图片
              </label>
            </div>
            <p class="mt-2 text-xs text-[var(--mt-text-dim)]">
              推送标签：已在 monbooru 的图片自动回写标签（每模型独立 source，互不合并）·
              推送图片：直传的原图会新建为 monbooru 图片，标签随图附带 ·
              只开「推送标签」时，未推送过的直传图标签仅保存在本机
            </p>
          </template>
        </NCard>
      </section>
    </div>

    <!-- Threshold dialog -->
    <NModal
      :show="dialogModel !== null"
      preset="card"
      :title="`阈值 · ${dialogModel?.name ?? ''}`"
      style="width: min(94vw, 560px)"
      @update:show="(v: boolean) => (!v ? (dialogModel = null) : undefined)"
    >
      <div v-if="dialogModel && dialogDraft" class="flex flex-col gap-3">
        <p class="text-xs text-[var(--mt-text-dim)]">
          留空的类别使用默认值；「禁用」的类别完全不输出。默认值来自模型目录（全局
          {{ dialogModel.default_threshold }}）。
        </p>
        <div class="flex items-center gap-3">
          <span class="w-20 shrink-0 text-xs text-[var(--mt-text-dim)]">全局阈值</span>
          <NInputNumber
            v-model:value="dialogDraft.global"
            size="small"
            :min="0.01"
            :max="0.99"
            :step="0.05"
            style="width: 130px"
          />
          <span class="text-xs text-[var(--mt-text-dim)]">
            输出类别 {{ dialogModel.emitted_categories.length }}
          </span>
        </div>
        <div class="flex flex-col gap-1.5">
          <div class="flex items-center gap-2 px-1 text-[11px] text-[var(--mt-text-dim)]">
            <span class="w-20 shrink-0">类别</span>
            <span class="w-[130px] shrink-0">阈值覆盖</span>
            <span class="w-[130px] shrink-0">标签上限</span>
            <span class="ml-auto">禁用</span>
          </div>
          <div
            v-for="(cat, catKey) in Object.keys(dialogDraft.cats)"
            :key="catKey"
            class="flex items-center gap-2 rounded-xl bg-[var(--mt-bg-soft)] px-2 py-1.5"
            :class="dialogDraft.cats[cat].off ? 'opacity-50' : ''"
          >
            <span class="w-20 shrink-0 truncate text-xs">{{ categoryZh[cat] ?? cat }}</span>
            <NInputNumber
              v-model:value="dialogDraft.cats[cat].thr"
              size="small"
              :min="0.01"
              :max="0.99"
              :step="0.05"
              :placeholder="String(dialogModel.effective.categories[cat] ?? dialogDraft.global)"
              style="width: 130px"
            />
            <NInputNumber
              v-model:value="dialogDraft.cats[cat].topk"
              size="small"
              :min="1"
              :max="100"
              :step="1"
              :placeholder="String(dialogModel.effective.top_k[cat] ?? '默认')"
              style="width: 130px"
            />
            <NSwitch
              v-model:value="dialogDraft.cats[cat].off"
              size="small"
              class="ml-auto"
            />
          </div>
        </div>
        <div class="flex justify-end gap-2 pt-1">
          <NButton size="small" @click="dialogModel = null">
            <template #icon><X :size="14" /></template>
            取消
          </NButton>
          <NButton size="small" type="primary" @click="saveThresholds">
            <template #icon><Check :size="14" /></template>
            保存阈值
          </NButton>
        </div>
      </div>
    </NModal>
  </NSpin>
</template>
