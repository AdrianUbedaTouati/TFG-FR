# Preview Undefined Error Fix

## Issue
When previewing the second layer of normalization, the system was throwing:
```
TypeError: Cannot read properties of undefined (reading 'original')
```

## Root Cause
The `showPreview` function was trying to access `preview.original` without checking if `preview` was null or undefined. This happened when:
1. The preview response didn't contain data for the specified column
2. The column name might have changed due to previous transformations

## Solution
Added proper null/undefined checks:

1. **Changed line 3512**: Added condition to check if `preview` exists before accessing its properties:
   ```javascript
   } else if (preview) {
   ```

2. **Added else clause (lines 3626-3639)**: Handle the case when preview is null:
   ```javascript
   } else {
       // No preview data available
       bodyHtml = `
           <div style="text-align: center; padding: 40px;">
               <i class="bi bi-exclamation-circle" style="font-size: 3rem; color: #ff6b6b;"></i>
               <h4 style="margin-top: 20px; color: #ff6b6b;">Sin datos de previsualización</h4>
               <p style="color: rgba(240, 249, 255, 0.7); margin-top: 10px;">
                   No se pudieron obtener datos de previsualización para la columna "${column}".
               </p>
               <button class="btn btn-secondary" onclick="closePreview()" style="margin-top: 20px;">
                   Cerrar
               </button>
           </div>
       `;
   }
   ```

3. **Added debug logging**: To help diagnose future issues:
   ```javascript
   console.log('Looking for column:', column, 'in preview:', response.preview);
   console.log('Preview data for column:', preview);
   ```

## Result
The system now gracefully handles cases where preview data is not available, showing a user-friendly message instead of crashing with a TypeError.