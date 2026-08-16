console.info("SW2.5 PDF.js diagnostics loaded");

document.addEventListener("webviewerloaded", () => {
  const app = window.PDFViewerApplication;

  app.initializedPromise.then(() => {
    const log = label => {
      const viewer = app.pdfViewer;

      console.log(label, {
        scale: viewer?.currentScaleValue,
        scrollMode: viewer?.scrollMode,
        spreadMode: viewer?.spreadMode,
      });
    };

    const events = [
      "pagesinit",
      "documentinit",
      "scalechanging",
      "scrollmodechanged",
      "spreadmodechanged",
      "updateviewarea",
      "documentloaded",
    ];

    for (const name of events) {
      app.eventBus.on(name, event => {
        console.log(`PDF.js event: ${name}`, event);
        log(`state after ${name}`);
      });
    }

    log("state after initializedPromise");
  });
});