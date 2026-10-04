import { createApp } from "vue";
import { createRouter, createWebHistory } from "vue-router";
import App from "./App.vue";
import UploadView from "./views/UploadView.vue";
import QueueView from "./views/QueueView.vue";
import SettingsView from "./views/SettingsView.vue";
import LoginView from "./views/LoginView.vue";
import { api } from "./api";
import "./style.css";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: UploadView, meta: { title: "打标" } },
    { path: "/queue", component: QueueView, meta: { title: "队列" } },
    { path: "/settings", component: SettingsView, meta: { title: "设置" } },
    { path: "/login", component: LoginView, meta: { title: "登录", bare: true } },
  ],
});

// A guard is only up once a password or token exists; then unauthenticated
// screens land on /login.
router.beforeEach(async (to) => {
  try {
    const state = await api.authState();
    if (to.path !== "/login" && state.guard_active && !state.authenticated) return "/login";
    if (to.path === "/login" && (!state.guard_active || state.authenticated)) return "/";
  } catch {
    /* backend unreachable: let the view render and show the error */
  }
  return true;
});

// Dark-first theme; the choice persists in localStorage (see theme.ts).
const theme = localStorage.getItem("mt-theme") ?? "dark";
document.documentElement.classList.toggle("light", theme === "light");

const app = createApp(App);
app.use(router);
app.mount("#app");

export { router };
