#!/usr/bin/env python3
"""M6 桌面常驻侧边栏 · 一体版（dsh web 并入侧边栏，2026-08-19）。

一个窗口双 Tab：「面板」= dashboard(:3091，夜报/告警/提案) ↔
「对话」= dsh web(:3090，鲸鱼娘+agent 对话)。鲸鱼娘与面板同栏共存。

窗口行为（EAC 风格参考）：
  - 宽 400px，贴屏幕右缘，避开顶部菜单栏（y=25，高=屏高-25）
  - 无边框（frameless）+ 置顶（on_top）；可拖动右缘调宽（resizable）
  - 顶部 Tab 条注入到两个页面（dashboard 和 dsh web 都注入），切换无感
  - Cmd+Shift+J 快速收起；底部工具条：刷新 / 收起

依赖：~/.workbuddy/binaries/python/envs/sidebar（pywebview）。
常驻：launchd plist = dsh/com.local-ai-agent.sidebar.plist。
"""
from __future__ import annotations

import sys

import webview  # pywebview

DASHBOARD_URL = "http://127.0.0.1:3091"
DSH_WEB_URL = "http://127.0.0.1:3090"
WIDTH = 400
MENU_BAR = 25  # macOS 顶部菜单栏高度，窗口避开它，否则底部按钮被压出屏外

# 顶部 Tab 条 + 底部工具条，注入到任意已加载页面（dashboard / dsh web 通用）
SHELL_JS = """
(function() {
  if (document.getElementById('__sidebar_tabs')) return;

  const css = (el, s) => { el.style.cssText = s; };
  const mkbtn = (label, fn, active) => {
    const b = document.createElement('button');
    b.textContent = label;
    css(b, 'border:1px solid #1e3a52;border-radius:6px;padding:3px 14px;'
      + 'cursor:pointer;font-size:12px;'
      + (active ? 'background:#4fb3ff;color:#06223a;font-weight:600;'
                : 'background:#10202f;color:#dceef9;'));
    b.onclick = fn;
    return b;
  };

  // 顶部 Tab 条
  const tabs = document.createElement('div');
  tabs.id = '__sidebar_tabs';
  css(tabs, 'position:fixed;top:0;left:0;right:0;z-index:2147483647;'
    + 'display:flex;gap:6px;justify-content:center;padding:6px;'
    + 'background:linear-gradient(135deg,#0d1f33 0%,#12395c 100%);'
    + 'border-bottom:1px solid #1e3a52;font:12px -apple-system,sans-serif;'
    + '-webkit-app-region:drag;');  // Tab 条可当标题栏拖窗口
  const onDash = location.port === '3091';
  tabs.appendChild(mkbtn('📊 面板', () => { if (!onDash) location.href = '__DASH__'; }, onDash));
  tabs.appendChild(mkbtn('🐳 对话', () => { if (onDash) location.href = '__DSH__'; }, !onDash));
  document.body.appendChild(tabs);
  document.body.style.paddingTop = '34px';

  // 底部工具条
  const bar = document.createElement('div');
  bar.id = '__sidebar_toolbar';
  css(bar, 'position:fixed;bottom:0;left:0;right:0;z-index:2147483647;'
    + 'display:flex;gap:8px;justify-content:center;padding:6px;'
    + 'background:rgba(10,20,32,.94);border-top:1px solid #1e3a52;'
    + 'font:12px -apple-system,sans-serif;');
  const hide = () => window.pywebview.api.hide();
  bar.appendChild(mkbtn('🔄 刷新', () => location.reload(), false));
  bar.appendChild(mkbtn('➖ 收起 (⌘⇧J)', hide, false));
  document.body.appendChild(bar);
  document.body.style.paddingBottom = '36px';

  // Cmd+Shift+J 快速收起
  document.addEventListener('keydown', (e) => {
    if (e.metaKey && e.shiftKey && e.key.toLowerCase() === 'j') hide();
  });
})();
""".replace("__DASH__", DASHBOARD_URL).replace("__DSH__", DSH_WEB_URL)


class SidebarApi:
    """暴露给 JS 的 API（window.pywebview.api.*）。"""

    def __init__(self):
        self.window = None

    def hide(self):
        if self.window:
            self.window.minimize()


class SidebarApp:
    def __init__(self, api: SidebarApi):
        self.api = api

    def on_loaded(self):
        # 每次页面加载完都注入 Tab 条（含 Tab 切换后的新页面）
        self.api.window.evaluate_js(SHELL_JS)


def main():
    api = SidebarApi()
    app = SidebarApp(api)
    screen = webview.screens[0]
    x = screen.width - WIDTH
    window = webview.create_window(
        "本地 AI",
        DASHBOARD_URL,
        js_api=api,
        width=WIDTH,
        height=screen.height - MENU_BAR,  # 避开菜单栏，底部按钮不再被压出屏外
        x=x,
        y=MENU_BAR,
        frameless=True,
        on_top=True,
        resizable=True,  # EAC 风格：允许拖右缘调宽
        min_size=(320, 480),
        background_color="#0a1420",
    )
    api.window = window
    window.events.loaded += app.on_loaded
    webview.start(debug=False)


if __name__ == "__main__":
    sys.exit(main())
