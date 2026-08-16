(() => {
  const STORAGE_KEY = "pdfjs.preferences";

  const defaults = {
    defaultZoomValue: "page-fit",
    scrollModeOnLoad: 0,
    spreadModeOnLoad: 2,
  };

  let preferences = {};

  try {
    const stored = localStorage.getItem(STORAGE_KEY);

    if (stored) {
      preferences = JSON.parse(stored) || {};
    }
  } catch (error) {
    console.warn(
      "PDF.js: failed to read stored preferences:",
      error
    );
    return;
  }

  let changed = false;

  for (const [name, value] of Object.entries(defaults)) {
    if (!Object.prototype.hasOwnProperty.call(preferences, name)) {
      preferences[name] = value;
      changed = true;
    }
  }

  if (changed) {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(preferences)
      );

      console.info(
        "PDF.js initial preferences installed:",
        defaults
      );
    } catch (error) {
      console.warn(
        "PDF.js: failed to install initial preferences:",
        error
      );
    }
  } else {
    console.info(
      "PDF.js existing preferences preserved:",
      {
        defaultZoomValue: preferences.defaultZoomValue,
        scrollModeOnLoad: preferences.scrollModeOnLoad,
        spreadModeOnLoad: preferences.spreadModeOnLoad,
      }
    );
  }
})();