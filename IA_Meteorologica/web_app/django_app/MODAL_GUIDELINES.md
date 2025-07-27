# Modal Guidelines - IMPORTANT

## ⚠️ NEVER USE BACKDROP IN MODALS

This application has a strict policy against modal backdrops to prevent gray overlay issues that block the interface.

## ✅ Correct Way to Create Modals

### 1. Use the Helper Functions (Recommended)

```javascript
// Option 1: Use showModal() for simple cases
showModal('myModalId');

// Option 2: Use createModal() when you need the modal instance
const modal = createModal(document.getElementById('myModal'));
modal.show();
```

### 2. If You Must Use Bootstrap Modal Directly

```javascript
// ALWAYS set backdrop: false
const modal = new bootstrap.Modal(modalElement, {
    backdrop: false,  // REQUIRED - prevents gray overlay
    keyboard: true,
    focus: true
});
```

## ❌ What NOT to Do

```javascript
// NEVER do this:
new bootstrap.Modal(element, { backdrop: 'static' });  // ❌ Creates gray overlay
new bootstrap.Modal(element, { backdrop: true });      // ❌ Creates gray overlay
new bootstrap.Modal(element);                          // ❌ Default backdrop might cause issues

// NEVER use these HTML attributes:
data-bs-backdrop="static"  // ❌
data-backdrop="static"     // ❌
```

## 📋 Checklist for New Modals

- [ ] Use `createModal()` or `showModal()` helper functions
- [ ] If using `new bootstrap.Modal()`, set `backdrop: false`
- [ ] Clean up modals properly with `modal.dispose()`
- [ ] Test that no gray overlay appears
- [ ] Check that clicking outside doesn't create issues

## 🔧 Automatic Protection

The application automatically:
1. Overrides Bootstrap Modal to force `backdrop: false`
2. Logs warnings if backdrop is detected
3. Hides all modal backdrops via CSS
4. Cleans up stray backdrops

## 💡 Tips

1. **Always clean up**: Call `cleanupModals()` before showing a new modal
2. **Dispose properly**: Use `modal.dispose()` when hiding modals programmatically
3. **Use helpers**: The helper functions handle all edge cases

## 🚨 If You See a Gray Overlay

1. Check for `backdrop: 'static'` or `backdrop: true` in your code
2. Look for `data-bs-backdrop` attributes in HTML
3. Use browser DevTools to inspect `.modal-backdrop` elements
4. Replace with the helper functions

## Example Implementation

```javascript
function showMyCustomModal() {
    // Clean up first
    cleanupModals();
    
    // Get your modal element
    const modalElement = document.getElementById('myCustomModal');
    
    // Create and show using helper
    const modal = createModal(modalElement);
    modal.show();
    
    // Handle closing
    modalElement.addEventListener('hidden.bs.modal', () => {
        modal.dispose();
        cleanupModals();
    });
}
```

Remember: **NO BACKDROPS, EVER!**