<script setup lang="ts">
import { computed } from "vue";
import { Tag } from "lucide-vue-next";
import type { Tag as TagT } from "../api";

const props = defineProps<{ tags: TagT[]; max?: number; copyable?: boolean }>();

// Danbooru-family chip colors.
const COLORS: Record<string, string> = {
  general: "#60a5fa",
  artist: "#f87171",
  copyright: "#c084fc",
  character: "#4ade80",
  meta: "#fbbf24",
  rating: "#facc15",
  medium: "#2dd4bf",
  person: "#f472b6",
  species: "#34d399",
  year: "#94a3b8",
};

// Rating marks get a badge treatment instead of a plain chip.
const RATING: Record<string, { symbol: string; label: string; color: string }> = {
  general: { symbol: "G", label: "一般", color: "#4ade80" },
  sensitive: { symbol: "S", label: "敏感", color: "#fbbf24" },
  questionable: { symbol: "Q", label: "疑似", color: "#fb923c" },
  explicit: { symbol: "E", label: "显式", color: "#f87171" },
};

function ratingOf(tag: TagT) {
  return RATING[tag.name.toLowerCase()] ?? null;
}

const shown = computed(() => (props.max ? props.tags.slice(0, props.max) : props.tags));
const rest = computed(() => (props.max ? Math.max(0, props.tags.length - props.max) : 0));
const pctOf = (tag: TagT) => Math.round((tag.confidence ?? 0) * 100);

function styleFor(tag: TagT) {
  const rating = ratingOf(tag);
  const color = rating ? rating.color : COLORS[tag.category] ?? "#94a3b8";
  return {
    color,
    borderColor: color + (rating ? "aa" : "55"),
    background: color + (rating ? "22" : "14"),
  };
}

async function copy(tag: TagT) {
  try {
    await navigator.clipboard.writeText(tag.name);
  } catch {
    /* clipboard unavailable */
  }
}
</script>

<template>
  <div class="flex flex-wrap gap-1.5">
    <span
      v-for="tag in shown"
      :key="tag.category + tag.name"
      class="relative inline-flex cursor-default items-center gap-1 overflow-hidden rounded-lg border px-2 pb-1 pt-0.5 text-xs font-medium"
      :style="styleFor(tag)"
      :title="`${tag.category} · ${pctOf(tag)}%`"
      @click="copyable && copy(tag)"
    >
      <template v-if="tag.category === 'rating'">
        <Tag v-if="!ratingOf(tag)" :size="10" />
        <b v-else class="text-[11px] leading-none">{{ ratingOf(tag)!.symbol }}</b>
      </template>
      {{ tag.name.replace(/_/g, " ") }}
      <span class="opacity-60">{{ pctOf(tag) }}</span>
      <!-- confidence underline -->
      <i
        class="absolute inset-x-1 bottom-0 h-[3px] rounded-full opacity-80"
        :style="{ width: pctOf(tag) + '%', background: (ratingOf(tag)?.color ?? COLORS[tag.category]) ?? '#94a3b8' }"
      />
    </span>
    <span v-if="rest > 0" class="self-center text-xs text-[var(--mt-text-dim)]">+{{ rest }}</span>
  </div>
</template>
