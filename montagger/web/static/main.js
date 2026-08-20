/* Small htmx additions. The UI is declarative; this file only fills gaps
   the attributes cannot express. */

document.addEventListener("htmx:beforeSwap", function (event) {
  if (event.detail.isError) {
    event.detail.shouldSwap = false;
    event.detail.shouldCancel = true;
  }
});

/* hx-confirm prompts go through an in-page <dialog> instead of the browser's
   native confirm(). A destructive action carries data-confirm-danger, which
   reddens the OK button and lands focus on cancel so an accidental Enter does
   not commit it. */

function confirmOpen() {
  var dlg = document.getElementById("confirm-dialog");
  return !!(dlg && dlg.open);
}

function showConfirm(message, onOk, okLabel, danger, detail) {
  var dlg = document.getElementById("confirm-dialog");
  if (!dlg) {
    if (window.confirm(detail ? message + "\n\n" + detail : message)) {
      onOk();
    }
    return;
  }
  document.getElementById("confirm-dialog-msg").textContent = message || "";
  var detailEl = document.getElementById("confirm-dialog-detail");
  if (detailEl) {
    detailEl.textContent = detail || "";
    detailEl.style.display = detail ? "" : "none";
  }
  var okBtn = document.getElementById("confirm-dialog-ok");
  var cancelBtn = document.getElementById("confirm-dialog-cancel");
  okBtn.textContent = okLabel || "ok";
  okBtn.classList.toggle("btn-danger", !!danger);
  var close = function () {
    dlg.close();
    okBtn.onclick = null;
    cancelBtn.onclick = null;
  };
  okBtn.onclick = function () {
    close();
    onOk();
  };
  cancelBtn.onclick = close;
  dlg.showModal();
  (danger ? cancelBtn : okBtn).focus();
}

// main.js runs from <head> before <body> exists, so listen on document -
// htmx:confirm bubbles up from the element carrying hx-confirm.
document.addEventListener("htmx:confirm", function (e) {
  if (!e.detail || !e.detail.question) {
    return;
  }
  e.preventDefault();
  var elt = e.detail.elt;
  var okLabel = elt && elt.dataset ? elt.dataset.confirmOk : "";
  var danger = !!(elt && elt.hasAttribute && elt.hasAttribute("data-confirm-danger"));
  // The attribute doubles as the warning subtext when it carries a value.
  var detail = elt && elt.getAttribute ? elt.getAttribute("data-confirm-danger") : "";
  showConfirm(e.detail.question, function () { e.detail.issueRequest(true); }, okLabel, danger, detail);
});