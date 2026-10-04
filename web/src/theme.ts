import { darkTheme, createDiscreteApi, type GlobalThemeOverrides } from "naive-ui";

export const themeOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: "#6366f1",
    primaryColorHover: "#818cf8",
    primaryColorPressed: "#4f46e5",
    primaryColorSuppl: "#6366f1",
    borderRadius: "10px",
  },
};

export const isDark = () => !document.documentElement.classList.contains("light");
export const naiveTheme = () => (isDark() ? darkTheme : undefined);

export const { message } = createDiscreteApi(["message"], {
  configProviderProps: {
    theme: naiveTheme(),
    themeOverrides,
  },
});

export function toggleTheme() {
  const dark = isDark();
  document.documentElement.classList.toggle("light", dark);
  localStorage.setItem("mt-theme", dark ? "light" : "dark");
}
