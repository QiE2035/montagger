<script setup lang="ts">
import { computed, ref } from "vue";
import { useRoute } from "vue-router";
import { NConfigProvider, NMessageProvider, NDialogProvider, darkTheme, zhCN, dateZhCN } from "naive-ui";
import { Tags, History, Settings, Moon, Sun } from "lucide-vue-next";
import { themeOverrides, toggleTheme } from "./theme";

const route = useRoute();

const dark = ref(document.documentElement.classList.contains("light") === false);

const nav = [
  { path: "/", label: "打标", icon: Tags },
  { path: "/queue", label: "队列", icon: History },
  { path: "/settings", label: "设置", icon: Settings },
];

const bare = computed(() => route.meta.bare === true);

function flipTheme() {
  toggleTheme();
  dark.value = !dark.value;
}
</script>

<template>
  <n-config-provider
    :theme="dark ? darkTheme : undefined"
    :theme-overrides="themeOverrides"
    :locale="zhCN"
    :date-locale="dateZhCN"
    style="height: 100%"
  >
    <n-message-provider placement="top">
      <n-dialog-provider>
        <div v-if="bare" style="height: 100%">
          <router-view />
        </div>
        <div v-else class="flex h-full flex-col md:flex-row">
          <!-- Desktop sidebar -->
          <aside
            class="hidden shrink-0 border-r border-[var(--mt-border)] bg-[var(--mt-bg-soft)] px-3 py-4 md:flex md:w-56 md:flex-col"
          >
            <div class="mb-8 flex items-center gap-2 px-2">
              <span class="grid size-8 place-items-center rounded-xl bg-[var(--mt-primary)] text-white">
                <Tags :size="18" />
              </span>
              <span class="text-lg font-semibold tracking-tight">montagger</span>
            </div>
            <nav class="flex flex-col gap-1">
              <router-link
                v-for="item in nav"
                :key="item.path"
                :to="item.path"
                class="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-[var(--mt-text-dim)] transition-colors hover:bg-[var(--mt-card)] hover:text-[var(--mt-text)]"
                active-class="!bg-[var(--mt-primary)] !text-white hover:!text-white"
              >
                <component :is="item.icon" :size="18" />
                {{ item.label }}
              </router-link>
            </nav>
            <button
              class="mt-auto flex items-center gap-3 self-start rounded-xl px-3 py-2 text-sm text-[var(--mt-text-dim)] hover:text-[var(--mt-text)]"
              @click="flipTheme"
            >
              <component :is="dark ? Sun : Moon" :size="18" />
              {{ dark ? "亮色" : "暗色" }}
            </button>
          </aside>

          <!-- Main column -->
          <div class="flex min-h-0 flex-1 flex-col">
            <header
              class="safe-top flex items-center gap-3 border-b border-[var(--mt-border)] bg-[var(--mt-bg-soft)]/80 px-4 py-3 backdrop-blur md:hidden"
            >
              <span class="grid size-7 place-items-center rounded-lg bg-[var(--mt-primary)] text-white">
                <Tags :size="15" />
              </span>
              <span class="font-semibold tracking-tight">montagger</span>
              <button class="ml-auto text-[var(--mt-text-dim)]" @click="flipTheme">
                <component :is="dark ? Sun : Moon" :size="20" />
              </button>
            </header>

            <main class="min-h-0 flex-1 overflow-y-auto pb-24 md:pb-6">
              <div class="mx-auto w-full max-w-5xl px-4 py-4 md:px-8 md:py-8">
                <router-view />
              </div>
            </main>

            <!-- Mobile bottom tabs -->
            <nav
              class="safe-bottom fixed inset-x-0 bottom-0 z-40 flex border-t border-[var(--mt-border)] bg-[var(--mt-bg-soft)]/95 backdrop-blur md:hidden"
            >
              <router-link
                v-for="item in nav"
                :key="item.path"
                :to="item.path"
                class="flex flex-1 flex-col items-center gap-1 py-2 text-[11px] text-[var(--mt-text-dim)]"
                active-class="text-[var(--mt-primary)]"
              >
                <component :is="item.icon" :size="22" />
                {{ item.label }}
              </router-link>
            </nav>
          </div>
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>
