<script setup lang="ts">
import { computed, h, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { Trash2, Copy, RefreshCw, XCircle, Search, ChevronLeft, ChevronRight } from "lucide-vue-next";
import { NButton, NTag, useDialog, useMessage } from "naive-ui";
import { api, connectEvents, type Job } from "../api";
import TagChips from "../components/TagChips.vue";

const message = useMessage();
const dialog = useDialog();

const PAGE_SIZE = 50;

// Server-paged history; SSE overlays live state onto the loaded page.
const rows = reactive<Map<string, Job>>(new Map());
const page = ref(1);
const total = ref(0);
const search = ref("");
const searchInput = ref("");
const loading = ref(true);
const selected = reactive(new Set<string>());
let es: EventSource | null = null;

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / PAGE_SIZE)));

async function refresh() {
  loading.value = true;
  try {
    const r = await api.jobs(PAGE_SIZE, (page.value - 1) * PAGE_SIZE, search.value);
    rows.clear();
    for (const job of r.jobs) rows.set(job.id, job);
    total.value = r.total;
    for (const id of [...selected]) if (!rows.has(id)) selected.delete(id);
  } catch (err) {
    message.error(String((err as Error).message));
  } finally {
    loading.value = false;
  }
}

function goTo(p: number) {
  page.value = Math.min(Math.max(1, p), totalPages.value);
  refresh();
}

function doSearch() {
  search.value = searchInput.value.trim();
  page.value = 1;
  refresh();
}

onMounted(() => {
  refresh();
  es = connectEvents((job) => {
    if (rows.has(job.id)) {
      rows.set(job.id, { ...rows.get(job.id), ...job });
    } else if (page.value === 1 && !search.value && (job.status === "queued" || job.status === "running")) {
      rows.set(job.id, job);
      total.value += 1;
    }
  });
});
onBeforeUnmount(() => es?.close());

// Per-FILE grouping: multi-model uploads keep one row per model inside the
// group card, never merged.
interface Group {
  key: string;
  name: string;
  jobs: Job[];
}

const activeGroups = computed<Group[]>(() => groupJobs(
  [...rows.values()].filter((j) => j.status === "queued" || j.status === "running"),
));
const historyGroups = computed<Group[]>(() => groupJobs(
  [...rows.values()].filter((j) => j.status !== "queued" && j.status !== "running"),
));

function groupJobs(jobs: Job[]): Group[] {
  const order: string[] = [];
  const map = new Map<string, Group>();
  for (const job of jobs) {
    const key = job.sha256 || job.id;
    let group = map.get(key);
    if (!group) {
      group = { key, name: job.filename, jobs: [] };
      map.set(key, group);
      order.push(key);
    }
    group.jobs.push(job);
  }
  return order.map((k) => map.get(k)!);
}

const statusLabel: Record<string, string> = {
  queued: "排队中",
  running: "推理中",
  done: "完成",
  error: "失败",
  lost: "已丢失",
  canceled: "已取消",
};

const STATUS_COLOR: Record<string, string> = {
  queued: "default",
  running: "info",
  done: "success",
  error: "error",
  lost: "warning",
  canceled: "default",
};

function copyTags(job: Job) {
  const tags = (job.tags ?? []).map((t) => t.name).join(" ");
  navigator.clipboard.writeText(tags).then(() => message.success("标签已复制"));
}

function toggleGroup(group: Group) {
  const all = group.jobs.every((j) => selected.has(j.id));
  for (const j of group.jobs) {
    if (all) selected.delete(j.id);
    else selected.add(j.id);
  }
}

function remove(job: Job) {
  api.deleteJob(job.id).then(refresh).catch((e) => message.error(e.message));
}

function cancel(job: Job) {
  api.cancelJob(job.id).then(refresh).catch((e) => message.error(e.message));
}

function batchDeleteSelected() {
  const ids = [...selected];
  if (!ids.length) return;
  api.batchJobs({ action: "delete", ids }).then((r) => {
    message.success(`已删除 ${r.affected} 条`);
    selected.clear();
    refresh();
  }).catch((e) => message.error(e.message));
}

function batchCancelSelected() {
  const ids = [...selected];
  if (!ids.length) return;
  api.batchJobs({ action: "cancel", ids }).then((r) => {
    message.success(`已取消 ${r.affected} 个`);
    refresh();
  }).catch((e) => message.error(e.message));
}

function cancelAllQueued() {
  api.batchJobs({ action: "cancel", all_queued: true }).then((r) => {
    message.success(`已取消 ${r.affected} 个排队任务`);
    refresh();
  }).catch((e) => message.error(e.message));
}

function clearFinished() {
  api.batchJobs({ action: "delete", all_done: true }).then((r) => {
    message.success(`已清除 ${r.affected} 条`);
    refresh();
  }).catch((e) => message.error(e.message));
}

function preview(job: Job) {
  dialog.create({
    title: job.filename,
    style: "width: min(92vw, 720px)",
    content: () =>
      h("div", { class: "flex flex-col gap-3" }, [
        h("img", {
          src: `/api/v1/jobs/${job.id}/preview`,
          class: "max-h-[50vh] w-full rounded-xl object-contain",
          onerror: (e: Event) => ((e.target as HTMLImageElement).style.display = "none"),
        }),
        h("div", { class: "text-xs text-[var(--mt-text-dim)]" }, [
          `${job.model} · ${job.width ?? "?"}×${job.height ?? "?"} · ${job.elapsed_ms ?? "?"}ms`,
        ]),
        h(TagChips, { tags: job.tags ?? [] }),
      ]),
  });
}
</script>

<template>
  <div class="flex flex-col gap-4">
    <div class="flex flex-wrap items-center justify-between gap-2 px-1">
      <h1 class="text-lg font-semibold">队列与历史</h1>
      <div class="flex flex-wrap items-center gap-2">
        <div class="flex items-center overflow-hidden rounded-full border border-[var(--mt-border)] bg-[var(--mt-card)]">
          <input
            v-model="searchInput"
            placeholder="搜索文件 / 模型 / 标签"
            class="w-40 bg-transparent px-3 py-1.5 text-xs outline-none md:w-52"
            @keydown.enter="doSearch"
          />
          <button class="px-2 text-[var(--mt-text-dim)] hover:text-[var(--mt-primary)]" @click="doSearch">
            <Search :size="14" />
          </button>
        </div>
        <n-button quaternary size="small" :disabled="loading" @click="refresh">
          <template #icon><RefreshCw :size="14" /></template>
          刷新
        </n-button>
        <n-button quaternary size="small" @click="cancelAllQueued">取消全部排队</n-button>
        <n-button quaternary size="small" @click="clearFinished">
          <template #icon><Trash2 :size="14" /></template>
          清除已完成
        </n-button>
      </div>
    </div>

    <!-- Batch bar -->
    <div
      v-if="selected.size"
      class="sticky top-2 z-10 flex flex-wrap items-center gap-2 rounded-2xl border border-[var(--mt-primary)]/40 bg-[var(--mt-card)] p-2.5 text-sm shadow-lg"
    >
      <span class="px-1">已选 {{ selected.size }} 项</span>
      <n-button size="tiny" secondary type="error" @click="batchDeleteSelected">删除所选</n-button>
      <n-button size="tiny" secondary @click="batchCancelSelected">取消所选</n-button>
      <n-button size="tiny" quaternary class="ml-auto" @click="selected.clear()">取消选择</n-button>
    </div>

    <!-- Active -->
    <div v-if="activeGroups.length" class="flex flex-col gap-2">
      <div
        v-for="group in activeGroups"
        :key="group.key"
        class="flex flex-wrap items-center gap-2 rounded-2xl border border-[var(--mt-border)] bg-[var(--mt-card)] p-3"
      >
        <input
          type="checkbox"
          class="accent-[var(--mt-primary)]"
          :checked="group.jobs.every((j) => selected.has(j.id))"
          @click.stop="toggleGroup(group)"
        />
        <RefreshCw :size="18" class="animate-spin text-[var(--mt-primary)]" />
        <span class="truncate text-sm">{{ group.name }}</span>
        <template v-for="job in group.jobs" :key="job.id">
          <n-tag size="small" round>{{ job.status === "running" ? "推理中" : "排队中" }}</n-tag>
          <n-tag size="small" round secondary>{{ job.model }}</n-tag>
          <n-button v-if="job.status === 'queued'" size="tiny" quaternary type="error" @click.stop="cancel(job)">
            <template #icon><XCircle :size="13" /></template>
          </n-button>
        </template>
      </div>
    </div>

    <!-- History, grouped per file -->
    <div class="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <div
        v-for="group in historyGroups"
        :key="group.key"
        class="flex cursor-pointer flex-col gap-2 rounded-2xl border border-[var(--mt-border)] bg-[var(--mt-card)] p-3"
        @click="preview(group.jobs[0])"
      >
        <div class="flex items-center gap-2">
          <input
            type="checkbox"
            class="accent-[var(--mt-primary)]"
            :checked="group.jobs.every((j) => selected.has(j.id))"
            @click.stop="toggleGroup(group)"
          />
          <span class="truncate text-sm font-medium">{{ group.name }}</span>
          <n-tag
            v-if="group.jobs.length > 1"
            size="small"
            round
            secondary
            class="ml-auto shrink-0"
          >{{ group.jobs.length }} 个模型</n-tag>
        </div>
        <div
          v-for="job in group.jobs"
          :key="job.id"
          class="flex flex-col gap-1.5 border-t border-[var(--mt-border)] pt-1.5 first:border-t-0 first:pt-0"
        >
          <div class="flex items-center gap-2">
            <n-tag size="small" round :type="(STATUS_COLOR[job.status] as any)" class="shrink-0">
              {{ statusLabel[job.status] ?? job.status }}
            </n-tag>
            <n-tag size="small" round secondary class="shrink-0">{{ job.model }}</n-tag>
            <div class="ml-auto flex gap-1" @click.stop>
              <n-button v-if="job.status === 'done'" size="tiny" quaternary @click="copyTags(job)">
                <template #icon><Copy :size="14" /></template>
              </n-button>
              <n-button size="tiny" quaternary type="error" @click="remove(job)">
                <template #icon><Trash2 :size="14" /></template>
              </n-button>
            </div>
          </div>
          <div v-if="job.status === 'error'" class="truncate text-xs text-red-400">{{ job.error }}</div>
          <div v-if="job.status === 'done'" class="flex items-center gap-2 text-xs text-[var(--mt-text-dim)]">
            {{ job.width }}×{{ job.height }} · {{ job.elapsed_ms }}ms · {{ job.provider }}
          </div>
          <TagChips v-if="job.status === 'done'" :tags="job.tags ?? []" :max="12" />
        </div>
      </div>
    </div>

    <!-- Pagination -->
    <div v-if="total > PAGE_SIZE" class="flex items-center justify-center gap-3 text-sm text-[var(--mt-text-dim)]">
      <n-button quaternary size="small" :disabled="page <= 1" @click="goTo(page - 1)">
        <template #icon><ChevronLeft :size="14" /></template>
      </n-button>
      <span>第 {{ page }} / {{ totalPages }} 页 · 共 {{ total }} 条</span>
      <n-button quaternary size="small" :disabled="page >= totalPages" @click="goTo(page + 1)">
        <template #icon><ChevronRight :size="14" /></template>
      </n-button>
    </div>

    <div v-if="!loading && rows.size === 0" class="py-16 text-center text-sm text-[var(--mt-text-dim)]">
      {{ search ? "没有匹配的任务" : "还没有任务 · 回到「打标」页推几张图试试" }}
    </div>
  </div>
</template>
