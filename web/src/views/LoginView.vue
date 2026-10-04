<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";
import { NButton, NCard, NInput } from "naive-ui";
import { Tags } from "lucide-vue-next";
import { api } from "../api";

const router = useRouter();
const password = ref("");
const error = ref("");
const busy = ref(false);

async function login() {
  busy.value = true;
  error.value = "";
  try {
    await api.login(password.value);
    router.push("/");
  } catch (err) {
    error.value = (err as Error).message || "登录失败";
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="grid h-full place-items-center bg-[var(--mt-bg)] px-6">
    <NCard class="w-full max-w-sm !rounded-3xl" :bordered="true">
      <div class="mb-6 flex flex-col items-center gap-2">
        <span class="grid size-12 place-items-center rounded-2xl bg-[var(--mt-primary)] text-white">
          <Tags :size="24" />
        </span>
        <span class="text-lg font-semibold tracking-tight">montagger</span>
      </div>
      <form class="flex flex-col gap-3" @submit.prevent="login">
        <NInput
          v-model:value="password"
          type="password"
          show-password-on="click"
          placeholder="访问密码"
          size="large"
          round
        />
        <NButton attr-type="submit" type="primary" size="large" round :loading="busy" block>
          进入
        </NButton>
        <p v-if="error" class="text-center text-xs text-red-400">{{ error }}</p>
      </form>
    </NCard>
  </div>
</template>
