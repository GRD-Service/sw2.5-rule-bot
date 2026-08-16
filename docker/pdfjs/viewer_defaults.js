console.info("SW2.5 PDF.js defaults loaded");

document.addEventListener("webviewerloaded", () => {
  const app = window.PDFViewerApplication;

  app.initializedPromise.then(() => {
    let applied = false;

    app.eventBus.on("pagesinit", async () => {
      if (applied) {
        return;
      }

      applied = true;

      const viewer = app.pdfViewer;
      const store = app.store;

      if (!viewer || !store) {
        return;
      }

      /*
       * ViewHistory に保存済み状態があるか確認。
       *
       * null を fallback にすることで、
       * 「未保存」と「保存済み」を区別する。
       */
      const history = await store.getMultiple({
        zoom: null,
        scrollMode: null,
        spreadMode: null,
      });

      console.info("SW2.5 existing history:", history);

      /*
       * 初めて開く PDF の項目だけ既定値を適用する。
       * 保存済みのユーザー表示状態は変更しない。
       */

      if (history.scrollMode === null) {
        viewer.scrollMode = 0;
      }

      if (history.spreadMode === null) {
        viewer.spreadMode = 2;
      }

      if (history.zoom === null) {
        /*
         * spreadMode 適用による再レイアウト後に
         * page-fit を設定する。
         */
        requestAnimationFrame(() => {
          viewer.currentScaleValue = "page-fit";

          console.info("SW2.5 defaults applied:", {
            scale: viewer.currentScaleValue,
            scrollMode: viewer.scrollMode,
            spreadMode: viewer.spreadMode,
          });
        });
      }
    });
  });
});