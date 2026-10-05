<script setup lang="ts">
import { computed, h, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { Trash2, Copy, RefreshCw, XCircle, Search, ChevronLeft, ChevronRight, ImageOff } from "lucide-vue-next";
import { NButton, NTag, useDialog, useMessage } from "naive-ui";
import { api, connectEvents, type Job } from "../api";
import TagChips from "../components/TagChips.vue";

const message = useMessage();
const dialog = useDialog();

const PAGE_SIZE = 50;

// Server-paged history, grouped by image server-side (total counts images,
// not jobs); SSE overlays live state onto the loaded page.
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
    const r = await api.jobs(PAGE_SIZE, (page.value - 1) * PAGE_SIZE, search.value, "", true);
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

// The page number doubles as an editable jump box.
const pageInput = ref("");
watch(page, (p) => (pageInput.value = String(p)), { immediate: true });

function submitPage() {
  const p = parseInt(pageInput.value, 10);
  if (Number.isNaN(p) || p === page.value) {
    pageInput.value = String(page.value);
    return;
  }
  goTo(p);
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
      return;
    }
    // The page is grouped by image: a new model's job for an image already
    // on this page just joins its group; a brand-new image lands only on
    // page 1 (and only while unfiltered), bumping the image total.
    const known =
      !!job.sha256 && [...rows.values()].some((j) => j.sha256 === job.sha256);
    if (known || (page.value === 1 && !search.value && (job.status === "queued" || job.status === "running"))) {
      rows.set(job.id, job);
      if (!known) total.value += 1;
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

// -- status vocabulary: color IS the status, words carry the detail -----------

const statusLabel: Record<string, string> = {
  queued: "排队中",
  running: "推理中",
  done: "完成",
  error: "失败",
  lost: "已丢失",
  canceled: "已取消",
};

const statusText: Record<string, string> = {
  queued: "text-[var(--mt-primary)]",
  running: "text-[var(--mt-primary)]",
  done: "text-green-400",
  error: "text-red-400",
  lost: "text-[var(--mt-text-dim)]",
  canceled: "text-[var(--mt-text-dim)]",
};

// The group's worst state drives its dot: failures first, then whatever is
// still moving, then the quiet terminal states.
const WORST_ORDER = ["error", "running", "queued", "lost", "canceled", "done"];

function worstStatus(group: Group): string {
  for (const s of WORST_ORDER) if (group.jobs.some((j) => j.status === s)) return s;
  return group.jobs[0]?.status ?? "done";
}

const dotColor: Record<string, string> = {
  queued: "bg-[var(--mt-primary)]",
  running: "bg-[var(--mt-primary)]",
  done: "bg-green-400",
  error: "bg-red-400",
  lost: "bg-[var(--mt-text-dim)]",
  canceled: "bg-[var(--mt-text-dim)]",
};

function humanSince(created?: string | number | null): string {
  if (created == null) return "";
  const ms = typeof created === "number" ? created * 1000 : new Date(`${created}Z`).getTime();
  const s = (Date.now() - ms) / 1000;
  if (!Number.isFinite(s) || s < 0) return "";
  if (s < 60) return "刚刚";
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`;
  return `${Math.floor(s / 86400)} 天前`;
}

function latestCreated(group: Group): string | number | undefined {
  const stamps = group.jobs.map((j) => j.created_at).filter((c) => c != null) as (string | number)[];
  if (!stamps.length) return undefined;
  stamps.sort((a, b) => (String(a) < String(b) ? -1 : String(a) > String(b) ? 1 : 0));
  return stamps[stamps.length - 1];
}

function groupDims(group: Group): string {
  const done = group.jobs.find((j) => j.width);
  return done?.width ? `${done.width}×${done.height}` : "";
}

// Previews come from montagger while the bytes are around, and from
// monbooru's thumbnail (persisted monbooru_id) once they are not.
function previewSrc(job: Job): string {
  return job.monbooru_id ? `/api/v1/jobs/${job.id}/remote-preview` : `/api/v1/jobs/${job.id}/preview`;
}

function hideBrokenImg(e: Event) {
  (e.target as HTMLImageElement).style.display = "none";
}

function toggleGroup(group: Group) {
  const all = group.jobs.every((j) => selected.has(j.id));
  for (const j of group.jobs) {
    if (all) selected.delete(j.id);
    else selected.add(j.id);
  }
}

// Page-scoped: only the loaded rows exist client-side, and refresh prunes
// the selection to whatever is still on screen.
function selectAll() {
  for (const job of rows.values()) selected.add(job.id);
}

function deselectAll() {
  selected.clear();
}

function copyModelTags(job: Job) {
  const tags = (job.tags ?? []).map((t) => t.name).join(" ");
  navigator.clipboard.writeText(tags).then(() => message.success(`${job.model} 标签已复制`));
}

function copyGroupTags(group: Group) {
  const text = group.jobs
    .filter((j) => j.status === "done" && j.tags?.length)
    .map((j) => `${j.model}: ${(j.tags ?? []).map((t) => t.name).join(" ")}`)
    .join("\n");
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => message.success("标签已复制"));
}

function cancel(job: Job) {
  api.cancelJob(job.id).then(refresh).catch((e) => message.error(e.message));
}

function removeGroup(group: Group) {
  api.batchJobs({ action: "delete", ids: group.jobs.map((j) => j.id) }).then((r) => {
    message.success(`已删除 ${r.affected} 条`);
    refresh();
  }).catch((e) => message.error(e.message));
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

function preview(group: Group) {
  dialog.create({
    title: group.name,
    style: "width: min(92vw, 720px)",
    content: () =>
      h("div", { class: "flex flex-col gap-3" }, [
        h("img", {
          src: previewSrc(group.jobs[0]),
          class: "max-h-[50vh] w-full rounded-xl object-contain",
          onerror: hideBrokenImg,
        }),
        ...group.jobs.map((job, i) =>
          h(
            "details",
            { key: job.id, open: i === 0, class: "rounded-xl bg-[var(--mt-bg-soft)] px-3 py-2" },
            [
              h("summary", { class: "flex cursor-pointer select-none items-center gap-2 text-xs [&::marker]:text-[var(--mt-text-dim)]" }, [
                h("span", { class: "font-medium" }, job.model),
                h("span", { class: statusText[job.status] }, statusLabel[job.status] ?? job.status),
                h("span", { class: "text-[var(--mt-text-dim)]" }, `${job.width ?? "?"}×${job.height ?? "?"} · ${job.elapsed_ms ?? "?"}ms`),
                h("button", {
                  class: "ml-auto rounded-md px-1.5 py-0.5 text-[11px] text-[var(--mt-text-dim)] hover:bg-[var(--mt-bg)] hover:text-[var(--mt-primary)]",
                  title: "复制该模型标签",
                  onClick: (e: Event) => {
                    e.stopPropagation();
                    e.preventDefault();
                    copyModelTags(job);
                  },
                }, "复制"),
              ]),
              h("div", { class: "mt-2 flex flex-col gap-1.5" }, [
                job.status === "error" ? h("div", { class: "text-xs text-red-400" }, job.error ?? "") : null,
                h(TagChips, { tags: job.tags ?? [] }),
              ]),
            ],
          ),
        ),
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
        <n-button quaternary size="small" @click="selectAll">全选</n-button>
        <n-button quaternary size="small" :disabled="!selected.size" @click="deselectAll">
          取消全选
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
      <n-button size="tiny" quaternary class="ml-auto" @click="deselectAll">取消选择</n-button>
    </div>

    <!-- Active -->
    <div v-if="activeGroups.length" class="flex flex-col gap-2">
      <div
        v-for="group in activeGroups"
        :key="group.key"
        class="group flex items-center gap-3 rounded-xl border border-[var(--mt-border)] bg-[var(--mt-card)] p-2.5"
      >
        <input
          type="checkbox"
          class="accent-[var(--mt-primary)]"
          :checked="group.jobs.every((j) => selected.has(j.id))"
          @click.stop="toggleGroup(group)"
        />
        <div class="relative h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-[var(--mt-bg-soft)]">
          <RefreshCw :size="18" class="absolute inset-0 m-auto animate-spin text-[var(--mt-primary)]" />
        </div>
        <div class="min-w-0 flex-1">
          <div class="flex items-center gap-2">
            <span class="truncate text-sm" :title="group.name">{{ group.name }}</span>
            <NTag v-if="group.jobs.length > 1" size="small" round secondary class="shrink-0">
              {{ group.jobs.length }} 模型
            </NTag>
          </div>
          <div class="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs">
            <span
              v-for="job in group.jobs"
              :key="job.id"
              :class="statusText[job.status]"
              :title="`${job.model} · ${statusLabel[job.status] ?? job.status}`"
            >{{ job.model }} {{ statusLabel[job.status] ?? job.status }}</span>
          </div>
        </div>
        <div class="flex shrink-0 gap-1">
          <n-button
            v-for="job in group.jobs.filter((j) => j.status === 'queued')"
            :key="job.id"
            size="tiny"
            quaternary
            type="error"
            :title="`取消 ${job.model}`"
            @click.stop="cancel(job)"
          >
            <template #icon><XCircle :size="13" /></template>
          </n-button>
        </div>
      </div>
    </div>

    <!-- History, one compact row per image -->
    <div class="grid grid-cols-1 gap-2 lg:grid-cols-2">
      <div
        v-for="group in historyGroups"
        :key="group.key"
        class="group flex cursor-pointer items-center gap-3 rounded-xl border border-[var(--mt-border)] bg-[var(--mt-card)] p-2.5"
        @click="preview(group)"
      >
        <input
          type="checkbox"
          class="accent-[var(--mt-primary)]"
          :checked="group.jobs.every((j) => selected.has(j.id))"
          @click.stop="toggleGroup(group)"
        />
        <div class="relative h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-[var(--mt-bg-soft)]">
          <ImageOff :size="16" class="absolute inset-0 m-auto text-[var(--mt-text-dim)]" />
          <img
            :src="previewSrc(group.jobs[0])"
            class="relative h-full w-full object-cover"
            loading="lazy"
            @error="hideBrokenImg"
          />
        </div>
        <div class="min-w-0 flex-1">
          <div class="flex items-center gap-2">
            <span class="flex min-w-0 items-center gap-1.5 text-sm" :title="group.name">
              <span class="h-2 w-2 shrink-0 rounded-full" :class="dotColor[worstStatus(group)]" />
              <span class="truncate">{{ group.name }}</span>
            </span>
            <NTag v-if="group.jobs.length > 1" size="small" round secondary class="shrink-0">
              {{ group.jobs.length }} 模型
            </NTag>
          </div>
          <div class="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs">
            <span
              v-for="job in group.jobs"
              :key="job.id"
              class="truncate"
              :class="statusText[job.status]"
              :title="`${job.model} · ${statusLabel[job.status] ?? job.status}`"
            >{{ job.model }} {{ statusLabel[job.status] ?? job.status }}</span>
          </div>
          <div
            class="mt-0.5 truncate text-[11px] text-[var(--mt-text-dim)]"
            :title="groupDims(group)"
          >
            {{ humanSince(latestCreated(group)) }}<template v-if="groupDims(group)"> · {{ groupDims(group) }}</template>
          </div>
        </div>
        <div class="hidden shrink-0 gap-1 group-hover:flex" @click.stop>
          <n-button
            v-if="group.jobs.some((j) => j.status === 'done')"
            size="tiny"
            quaternary
            title="复制标签"
            @click.stop="copyGroupTags(group)"
          >
            <template #icon><Copy :size="13" /></template>
          </n-button>
          <n-button size="tiny" quaternary type="error" title="删除该图的所有任务" @click.stop="removeGroup(group)">
            <template #icon><Trash2 :size="13" /></template>
          </n-button>
        </div>
      </div>
    </div>

    <!-- Pagination -->
    <div v-if="total > PAGE_SIZE" class="flex items-center justify-center gap-3 text-sm text-[var(--mt-text-dim)]">
      <n-button quaternary size="small" :disabled="page <= 1" @click="goTo(page - 1)">
        <template #icon><ChevronLeft :size="14" /></template>
      </n-button>
      <span class="flex items-center gap-1">
        第
        <input
          v-model="pageInput"
          @keydown.enter="submitPage"
          @blur="submitPage"
          class="w-12 rounded-lg border border-[var(--mt-border)] bg-[var(--mt-card)] px-1 py-0.5 text-center text-xs outline-none focus:border-[var(--mt-primary)]"
        />
        / {{ totalPages }} 页 · 共 {{ total }} 张图
      </span>
      <n-button quaternary size="small" :disabled="page >= totalPages" @click="goTo(page + 1)">
        <template #icon><ChevronRight :size="14" /></template>
      </n-button>
    </div>

    <div v-if="!loading && rows.size === 0" class="py-16 text-center text-sm text-[var(--mt-text-dim)]">
      {{ search ? "没有匹配的任务" : "还没有任务 · 回到「打标」页推几张图试试" }}
    </div>
  </div>
</template>
