document.addEventListener("webviewerloaded", () => {
  const app = window.PDFViewerApplication;

  if (!app) {
    console.error("PDF.js: PDFViewerApplication is not available.");
    return;
  }

  app.initializedPromise.then(() => {
    app.eventBus.on("documentinit", async () => {
      const viewer = app.pdfViewer;
      const store = app.store;

      if (!viewer || !store) {
        return;
      }

      try {
        /*
         * PDF.js stores per-document display state in ViewHistory.
         *
         * By requesting null as the fallback we can distinguish:
         *
         *   null       = user has never stored this setting
         *   non-null   = PDF.js has previous view history
         */
        const history = await store.getMultiple({
          zoom: null,
          scrollMode: null,
          spreadMode: null,
        });

        console.info(
          "PDF.js existing view history:",
          history
        );

        /*
         * Only apply our defaults when that particular value has
         * never been stored for this PDF.
         *
         * This means subsequent user changes are preserved.
         */

        if (history.scrollMode === null) {
          viewer.scrollMode = 0;
        }

        if (history.spreadMode === null) {
          viewer.spreadMode = 2;
        }

        if (history.zoom === null) {
          viewer.currentScaleValue = "page-fit";
        }

        console.info(
          "PDF.js effective defaults:",
          {
            zoom: viewer.currentScaleValue,
            scrollMode: viewer.scrollMode,
            spreadMode: viewer.spreadMode,
          }
        );
      } catch (error) {
        console.error(
          "PDF.js: failed to apply initial view defaults:",
          error
        );
      }
    });
  });
});