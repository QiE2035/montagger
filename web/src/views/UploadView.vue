<script setup lang="ts">
import { h, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { CloudUpload, Camera, Copy, Check, Image as ImageIcon, Loader2 } from "lucide-vue-next";
import { NButton, NProgress, useDialog, useMessage } from "naive-ui";
import { api, connectEvents, uploadFile, type Job } from "../api";
import TagChips from "../components/TagChips.vue";

/** One card per FILE; multi-model uploads hold one Job per model. */
interface Item {
  key: number;
  file: File;
  blobUrl: string;
  sent: number;
  total: number;
  uploading: boolean;
  uploadError: string;
  jobs: Job[];
  copiedKey: string;
}

const message = useMessage();
const dialog = useDialog();

const items = reactive<Item[]>([]);
const fileInput = ref<HTMLInputElement | null>(null);
const cameraInput = ref<HTMLInputElement | null>(null);
const dragging = ref(false);
const modelLabel = ref<string | null>(null);
let es: EventSource | null = null;
let keySeq = 0;

onMounted(async () => {
  try {
    const models = await api.models();
    const names = models.default_models.length ? models.default_models : [models.default];
    modelLabel.value = names.join(" + ");
  } catch {
    /* settings may need auth; the tag call still works */
  }
  es = connectEvents((job) => {
    const item = items.find((i) => i.jobs.some((j) => j.id === job.id));
    if (item) item.jobs = item.jobs.map((j) => (j.id === job.id ? job : j));
  });
});

onBeforeUnmount(() => es?.close());

function pickFiles(files: FileList | null) {
  if (!files) return;
  for (const file of Array.from(files)) {
    if (!file.type.startsWith("image/")) continue;
    const item: Item = {
      key: ++keySeq,
      file,
      blobUrl: URL.createObjectURL(file),
      sent: 0,
      total: file.size,
      uploading: true,
      uploadError: "",
      jobs: [],
      copiedKey: "",
    };
    items.unshift(item);
    uploadFile(file, null, (sent, total) => {
      item.sent = sent;
      item.total = total;
    })
      .then((body) => {
        item.jobs = Array.isArray(body) ? body : [body];
        item.uploading = false;
      })
      .catch((err) => {
        item.uploading = false;
        item.uploadError = err?.message ?? "上传失败";
      });
  }
}

function onDrop(e: DragEvent) {
  dragging.value = false;
  pickFiles(e.dataTransfer?.files ?? null);
}

const pct = (item: Item) =>
  item.total > 0 ? Math.round(((item.sent / item.total) * 100) | 0) : 0;

const statusLabel: Record<string, string> = {
  queued: "排队中",
  running: "推理中",
  done: "完成",
  error: "失败",
  lost: "已丢失",
  canceled: "已取消",
};

function statusOf(item: Item): string {
  if (item.uploadError) return "上传失败";
  if (item.uploading) return `上传中 ${pct(item)}%`;
  if (!item.jobs.length) return "入队中";
  if (item.jobs.length === 1) return statusLabel[item.jobs[0].status] ?? item.jobs[0].status;
  const done = item.jobs.filter((j) => j.status === "done").length;
  if (done === item.jobs.length) return `${done} 个模型完成`;
  if (item.jobs.some((j) => j.status === "running")) return `推理中 ${done}/${item.jobs.length}`;
  if (item.jobs.some((j) => j.status === "error")) return `${done}/${item.jobs.length} 完成`;
  return `排队中 ${done}/${item.jobs.length}`;
}

const busy = (item: Item) =>
  item.uploading ||
  !item.jobs.length ||
  item.jobs.some((j) => j.status === "queued" || j.status === "running");

function copyTags(item: Item, job?: Job) {
  const target = job ?? item.jobs.find((j) => j.status === "done");
  const tags = (target?.tags ?? []).map((t) => t.name).join(" ");
  navigator.clipboard
    .writeText(tags)
    .then(() => {
      item.copiedKey = target?.id ?? "";
      setTimeout(() => (item.copiedKey = ""), 1500);
    })
    .catch(() => message.error("剪贴板不可用"));
}

/** Copy grouped by category: `character: a, b` per line, general bare. */
function copyGrouped(job: Job) {
  const byCat = new Map<string, string[]>();
  for (const t of job.tags ?? []) {
    const list = byCat.get(t.category) ?? [];
    list.push(t.name);
    byCat.set(t.category, list);
  }
  const lines = [...byCat.entries()]
    .filter(([, names]) => names.length)
    .map(([category, names]) => `${category}: ${names.join(", ")}`);
  navigator.clipboard.writeText(lines.join("\n")).then(
    () => message.success("已按类别复制"),
    () => message.error("剪贴板不可用"),
  );
}

/** Download `category:name` per line — the monbooru-ready form. */
function downloadTxt(job: Job) {
  const lines = (job.tags ?? []).map((t) => `${t.category}:${t.name}`);
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = (job.filename.replace(/\.[^.]+$/, "") || "tags") + `.${job.model}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

function clearAll() {
  for (const item of items) URL.revokeObjectURL(item.blobUrl);
  items.length = 0;
}

function preview(item: Item) {
  if (!item.jobs.length) return;
  dialog.create({
    title: item.file.name,
    style: "width: min(92vw, 720px)",
    content: () =>
      h("div", { class: "flex flex-col gap-4" }, [
        h("img", { src: item.blobUrl, class: "max-h-[50vh] w-full rounded-xl object-contain" }),
        ...item.jobs.map((job) =>
          h("div", { class: "flex flex-col gap-2" }, [
            h("div", { class: "flex items-center gap-2" }, [
              h(
                "span",
                { class: "text-xs font-medium text-[var(--mt-text-dim)]" },
                `${job.model} · ${statusLabel[job.status] ?? job.status}`,
              ),
              job.status === "done"
                ? h("div", { class: "ml-auto flex gap-2" }, [
                    h(
                      "button",
                      {
                        class: "text-xs text-[var(--mt-primary)]",
                        onClick: () => copyTags(item, job),
                      },
                      item.copiedKey === job.id ? "已复制" : "复制",
                    ),
                    h(
                      "button",
                      { class: "text-xs text-[var(--mt-primary)]", onClick: () => copyGrouped(job) },
                      "按类复制",
                    ),
                    h(
                      "button",
                      { class: "text-xs text-[var(--mt-primary)]", onClick: () => downloadTxt(job) },
                      "txt",
                    ),
                  ])
                : null,
            ]),
            job.status === "done"
              ? h(TagChips, { tags: job.tags ?? [] })
              : job.status === "error"
                ? h("div", { class: "text-xs text-red-400" }, job.error ?? "")
                : null,
          ]),
        ),
      ]),
  });
}
</script>

<template>
  <div class="flex flex-col gap-4">
    <!-- Drop zone -->
    <div
      class="rounded-3xl border-2 border-dashed p-6 text-center transition-colors md:p-10"
      :class="dragging ? 'border-[var(--mt-primary)] bg-[var(--mt-primary)]/10' : 'border-[var(--mt-border)] bg-[var(--mt-bg-soft)]'"
      @dragover.prevent="dragging = true"
      @dragleave="dragging = false"
      @drop.prevent="onDrop"
    >
      <CloudUpload :size="44" class="mx-auto mb-3 text-[var(--mt-primary)]" />
      <p class="mb-1 text-base font-medium">拖放图片到这里，或</p>
      <p class="mb-4 text-sm text-[var(--mt-text-dim)]">原图直接推送到这台 PC，由 {{ modelLabel ?? "默认模型" }} 打标</p>
      <div class="flex flex-wrap justify-center gap-2">
        <n-button type="primary" size="large" round @click="fileInput?.click()">
          <template #icon><ImageIcon :size="18" /></template>
          选择图片
        </n-button>
        <n-button size="large" round @click="cameraInput?.click()">
          <template #icon><Camera :size="18" /></template>
          拍照
        </n-button>
      </div>
      <input
        ref="fileInput"
        type="file"
        accept="image/*"
        multiple
        class="hidden"
        @change="pickFiles(($event.target as HTMLInputElement).files); ($event.target as HTMLInputElement).value = ''"
      />
      <input
        ref="cameraInput"
        type="file"
        accept="image/*"
        capture="environment"
        class="hidden"
        @change="pickFiles(($event.target as HTMLInputElement).files); ($event.target as HTMLInputElement).value = ''"
      />
    </div>

    <!-- Upload list -->
    <div v-if="items.length" class="flex items-center justify-between px-1">
      <span class="text-sm text-[var(--mt-text-dim)]">{{ items.length }} 张</span>
      <n-button quaternary size="small" @click="clearAll">清空</n-button>
    </div>

    <div class="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <div
        v-for="item in items"
        :key="item.key"
        class="flex cursor-pointer gap-3 rounded-2xl border border-[var(--mt-border)] bg-[var(--mt-card)] p-3"
        @click="preview(item)"
      >
        <img :src="item.blobUrl" class="size-20 shrink-0 rounded-xl object-cover" />
        <div class="flex min-w-0 flex-1 flex-col gap-1.5">
          <div class="flex items-center gap-2">
            <span class="truncate text-sm font-medium">{{ item.file.name }}</span>
            <span
              class="ml-auto shrink-0 text-xs"
              :class="item.jobs.some((j) => j.status === 'done') && !busy(item) ? 'text-green-400' : item.uploadError ? 'text-red-400' : 'text-[var(--mt-text-dim)]'"
            >
              <Loader2 v-if="busy(item)" :size="13" class="inline animate-spin" />
              {{ statusOf(item) }}
            </span>
          </div>
          <n-progress
            v-if="item.uploading"
            type="line"
            :percentage="pct(item)"
            :show-indicator="false"
            :height="5"
            border-radius="3px"
          />
          <div v-if="item.uploadError" class="text-xs text-red-400">{{ item.uploadError }}</div>
          <div v-for="job in item.jobs" :key="job.id" class="flex flex-col gap-1">
            <div class="flex items-center gap-1.5 text-xs">
              <span
                class="shrink-0 rounded-full px-1.5 py-0.5"
                :class="{
                  'bg-green-500/15 text-green-400': job.status === 'done',
                  'bg-red-500/15 text-red-400': job.status === 'error' || job.status === 'lost' || job.status === 'canceled',
                  'bg-[var(--mt-primary)]/15 text-[var(--mt-primary)]': job.status === 'running' || job.status === 'queued',
                }"
              >{{ statusLabel[job.status] ?? job.status }}</span>
              <span class="truncate text-[var(--mt-text-dim)]">{{ job.model }}</span>
            </div>
            <div v-if="job.status === 'error'" class="truncate text-xs text-red-400">{{ job.error }}</div>
            <TagChips v-if="job.status === 'done'" :tags="job.tags ?? []" :max="10" />
          </div>
          <div v-if="item.jobs.some((j) => j.status === 'done')" class="mt-auto flex gap-1 self-end">
            <n-button size="tiny" quaternary @click.stop="copyTags(item)">
              <template #icon>
                <Check v-if="item.copiedKey" :size="14" class="text-green-400" />
                <Copy v-else :size="14" />
              </template>
              {{ item.copiedKey ? "已复制" : "复制标签" }}
            </n-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
