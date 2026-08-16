document.addEventListener("webviewerloaded", () => {
  const options = window.PDFViewerApplicationOptions;
  const app = window.PDFViewerApplication;

  if (!options || !app) {
    console.error("PDF.js application objects are not available.");
    return;
  }

  const defaults = {
    defaultZoomValue: "page-fit",
    scrollModeOnLoad: 0,
    spreadModeOnLoad: 2,
  };

  options.set("defaultZoomValue", defaults.defaultZoomValue);
  options.set("scrollModeOnLoad", defaults.scrollModeOnLoad);
  options.set("spreadModeOnLoad", defaults.spreadModeOnLoad);

  let storedPreferences = {};

  try {
    storedPreferences = JSON.parse(
      localStorage.getItem("pdfjs.preferences") || "{}"
    );
  } catch (error) {
    console.warn(
      "Could not read PDF.js preferences:",
      error
    );
  }

  app.initializedPromise.then(() => {
    app.eventBus.on("pagesinit", () => {
      // URLで zoom が指定されていれば最優先
      const hash = new URLSearchParams(
        window.location.hash.substring(1)
      );

      if (hash.has("zoom")) {
        return;
      }

      // ユーザーがズーム設定を保存済みなら尊重
      if (
        Object.prototype.hasOwnProperty.call(
          storedPreferences,
          "defaultZoomValue"
        )
      ) {
        return;
      }

      // 初回ユーザーのみ page-fit
      if (app.pdfViewer) {
        app.pdfViewer.currentScaleValue =
          defaults.defaultZoomValue;
      }
    });
  });

  console.info(
    "PDF.js defaults:",
    defaults,
    "stored preferences:",
    storedPreferences
  );
});