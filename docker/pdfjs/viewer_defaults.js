document.addEventListener("webviewerloaded", () => {
  const options = window.PDFViewerApplicationOptions;

  if (!options) {
    console.error(
      "PDF.js: PDFViewerApplicationOptions is not available."
    );
    return;
  }

  // SW2.5 Rule Bot default viewer settings.
  //
  // These are defaults only.
  // Stored PDF.js user preferences are loaded afterwards and therefore
  // override these values when the user has changed them.

  // ページ全体を表示
  options.set("defaultZoomValue", "page-fit");

  // 縦スクロール
  options.set("scrollModeOnLoad", 0);

  // 偶数ページ見開き
  // 1ページ目を単独、その後 2-3, 4-5, ...
  options.set("spreadModeOnLoad", 2);

  console.info(
    "PDF.js defaults:",
    "zoom=page-fit",
    "scrollMode=0",
    "spreadMode=2"
  );
});