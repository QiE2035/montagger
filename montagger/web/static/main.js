/* Small htmx additions. The UI is declarative; this file only fills gaps
   the attributes cannot express. */

document.addEventListener("htmx:beforeSwap", function (event) {
  if (event.detail.isError) {
    event.detail.shouldSwap = false;
    event.detail.shouldCancel = true;
  }
});