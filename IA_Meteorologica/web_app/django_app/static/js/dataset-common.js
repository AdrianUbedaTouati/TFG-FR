// Funciones comunes para el manejo de datasets

// Función para renombrar una columna del dataset
function renameColumn(columnName) {
    if (!selectedDataset) return;
    
    // Limpiar cualquier estado de modal anterior
    document.querySelectorAll('[aria-hidden="true"]').forEach(el => {
        el.removeAttribute('aria-hidden');
    });
    
    // Remover cualquier backdrop residual
    document.querySelectorAll('.modal-backdrop').forEach(backdrop => backdrop.remove());
    
    // Asegurar que el body no tenga clases de modal
    document.body.classList.remove('modal-open');
    document.body.style.removeProperty('overflow');
    document.body.style.removeProperty('padding-right');
    
    // Cerrar cualquier modal de Bootstrap abierto
    const openModals = document.querySelectorAll('.modal.show');
    openModals.forEach(modal => {
        const bsModal = bootstrap.Modal.getInstance(modal);
        if (bsModal) {
            bsModal.hide();
        }
    });
    
    // Pequeña demora para asegurar que todo se limpie
    setTimeout(() => {
        Swal.fire({
        title: 'Renombrar columna',
        text: `Nombre actual: ${columnName}`,
        input: 'text',
        inputLabel: 'Nuevo nombre:',
        inputValue: columnName,
        inputPlaceholder: 'Ingrese el nuevo nombre de la columna',
        icon: 'question',
        showCancelButton: true,
        confirmButtonColor: '#10b981',
        cancelButtonColor: '#6b7280',
        confirmButtonText: 'Renombrar',
        cancelButtonText: 'Cancelar',
        background: '#0a0e27',
        color: '#f0f9ff',
        allowOutsideClick: false,
        allowEscapeKey: true,
        allowEnterKey: true,
        stopKeydownPropagation: false,
        customClass: {
            popup: 'swal-custom-popup',
            title: 'swal-custom-title',
            htmlContainer: 'swal-custom-text',
            confirmButton: 'swal-custom-button',
            cancelButton: 'btn btn-secondary',
            input: 'swal2-input-custom'
        },
        didOpen: () => {
            // Asegurar que no haya conflictos de aria-hidden
            const swalContainer = document.querySelector('.swal2-container');
            if (swalContainer) {
                swalContainer.removeAttribute('aria-hidden');
            }
            
            const input = Swal.getInput();
            if (input) {
                // Aplicar estilos al input
                input.style.backgroundColor = '#1a1e3a';
                input.style.color = '#f0f9ff';
                input.style.borderColor = '#2d3561';
                input.style.padding = '8px 12px';
                
                // Asegurar que el input sea accesible
                input.removeAttribute('aria-hidden');
                input.removeAttribute('disabled');
                input.removeAttribute('readonly');
                
                // Forzar el focus con múltiples intentos
                const focusInput = () => {
                    input.focus();
                    input.select();
                    
                    // Verificar si el input tiene focus
                    if (document.activeElement !== input) {
                        console.log('Intentando enfocar nuevamente...');
                        setTimeout(focusInput, 50);
                    } else {
                        console.log('Input enfocado correctamente');
                    }
                };
                
                // Iniciar el proceso de focus
                setTimeout(focusInput, 100);
            }
        },
        inputValidator: (value) => {
            if (!value || !value.trim()) {
                return 'Por favor ingrese un nombre válido';
            }
            if (value.trim() === columnName) {
                return 'El nuevo nombre debe ser diferente al actual';
            }
        },
        preConfirm: (newName) => {
            return newName.trim();
        }
    }).then((result) => {
        if (result.isConfirmed) {
            const newName = result.value;
            showLoading('Renombrando columna...', `Cambiando "${columnName}" a "${newName}"`);
            
            fetch(`/api/datasets/${selectedDataset.id}/rename-column/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    old_name: columnName,
                    new_name: newName
                })
            })
            .then(response => {
                hideLoading();
                if (response.ok) {
                    return response.json();
                } else {
                    return response.json().then(data => {
                        throw new Error(data.error || `HTTP error! status: ${response.status}`);
                    });
                }
            })
            .then(data => {
                hideLoading();
                showNotification('¡Éxito!', `La columna "${columnName}" ha sido renombrada a "${newName}"`, 'success');
                
                // Recargar la página después de un momento
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            })
            .catch(error => {
                hideLoading();
                console.error('Error al renombrar columna:', error);
                showNotification('Error', 'Error al renombrar la columna: ' + error.message, 'error');
            });
        }
    });
    }, 100); // Fin del setTimeout
}

// Función para eliminar una columna
function deleteColumn(columnName) {
    if (!selectedDataset) return;
    
    Swal.fire({
        title: '¿Estás seguro?',
        text: `¿Deseas eliminar la columna "${columnName}" del dataset?`,
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#ef4444',
        cancelButtonColor: '#6b7280',
        confirmButtonText: 'Sí, eliminar',
        cancelButtonText: 'Cancelar',
        background: '#0a0e27',
        color: '#f0f9ff',
        customClass: {
            popup: 'swal-custom-popup',
            title: 'swal-custom-title',
            htmlContainer: 'swal-custom-text',
            confirmButton: 'btn btn-danger',
            cancelButton: 'btn btn-secondary'
        }
    }).then((result) => {
        if (result.isConfirmed) {
            showLoading('Eliminando columna...', `Eliminando la columna "${columnName}" del dataset`);
            
            fetch(`/api/datasets/${selectedDataset.id}/delete-column/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    column_name: columnName
                })
            })
            .then(response => {
                hideLoading();
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
            })
            .then(data => {
                hideLoading();
                showNotification('¡Éxito!', `La columna "${columnName}" ha sido eliminada`, 'success');
                
                // Recargar la página después de un momento
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            })
            .catch(error => {
                hideLoading();
                console.error('Error al eliminar columna:', error);
                showNotification('Error', 'Error al eliminar la columna: ' + error.message, 'error');
            });
        }
    });
}

// Función para mostrar valores únicos
function showUniqueValues(columnName) {
    const data = window.currentDatasetData;
    if (!data || !data.stats || !data.stats[columnName]) {
        showNotification('Error', 'No hay datos disponibles para esta columna', 'error');
        return;
    }
    
    const colStats = data.stats[columnName];
    
    Swal.fire({
        title: `Valores únicos - ${columnName}`,
        html: `
            <div class="text-start">
                <p><strong>Total valores únicos:</strong> ${colStats.unique_count || 0}</p>
                <p><strong>Valores nulos:</strong> ${colStats.null_count || 0}</p>
                <div id="uniqueValuesLoading" class="text-center">
                    <div class="spinner-border spinner-border-sm" role="status">
                        <span class="visually-hidden">Cargando...</span>
                    </div>
                </div>
                <div id="uniqueValuesContent" style="max-height: 400px; overflow-y: auto;"></div>
            </div>
        `,
        width: '800px',
        background: '#0a0e27',
        color: '#f0f9ff',
        showCloseButton: true,
        showConfirmButton: false,
        customClass: {
            popup: 'swal-custom-popup',
            title: 'swal-custom-title',
            htmlContainer: 'swal-custom-text'
        },
        didOpen: () => {
            // Cargar valores únicos
            fetch(`/api/datasets/${selectedDataset.id}/columns/${encodeURIComponent(columnName)}/`)
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('uniqueValuesContent');
                    const loading = document.getElementById('uniqueValuesLoading');
                    loading.style.display = 'none';
                    
                    if (data.frequency_data && data.frequency_data.length > 0) {
                        let html = `
                            <table class="table table-sm table-dark table-hover">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Valor</th>
                                        <th>Frecuencia</th>
                                        <th>Porcentaje</th>
                                    </tr>
                                </thead>
                                <tbody>
                        `;
                        
                        data.frequency_data.slice(0, 50).forEach((item, index) => {
                            html += `
                                <tr>
                                    <td>${index + 1}</td>
                                    <td><code>${item.value}</code></td>
                                    <td>${item.count.toLocaleString()}</td>
                                    <td>
                                        <div class="progress" style="height: 20px;">
                                            <div class="progress-bar bg-info" role="progressbar" 
                                                 style="width: ${item.percentage}%">
                                                ${item.percentage.toFixed(2)}%
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                            `;
                        });
                        
                        html += `
                                </tbody>
                            </table>
                        `;
                        
                        if (data.frequency_data.length > 50) {
                            html += `<p class="text-muted text-center">Mostrando los primeros 50 valores de ${data.frequency_data.length} totales</p>`;
                        }
                        
                        container.innerHTML = html;
                    } else {
                        container.innerHTML = '<p class="text-muted">No se encontraron datos de frecuencia</p>';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('uniqueValuesLoading').style.display = 'none';
                    document.getElementById('uniqueValuesContent').innerHTML = 
                        '<p class="text-danger">Error al cargar los valores únicos</p>';
                });
        }
    });
}

// Función para mostrar outliers
function showOutliers(columnName) {
    showNotification('Info', 'Generando análisis de outliers...', 'info');
    
    fetch(`/api/datasets/${selectedDataset.id}/columns/${encodeURIComponent(columnName)}/analysis/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({
            analysis_type: 'outlier_map'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        const outlierInfo = data.outlier_info || {};
        
        Swal.fire({
            title: `Análisis de Outliers - ${columnName}`,
            html: `
                <div class="text-start">
                    <div class="alert alert-info">
                        <strong>Outliers detectados:</strong> ${outlierInfo.total_outliers || 0} 
                        (${outlierInfo.outlier_percentage?.toFixed(2) || '0.00'}%)
                        <br>
                        <small>Límites: [${outlierInfo.lower_bound?.toFixed(2) || 'N/A'}, ${outlierInfo.upper_bound?.toFixed(2) || 'N/A'}]</small>
                    </div>
                    <img src="${data.plot || data.image}" class="img-fluid" alt="Outlier Map">
                </div>
            `,
            width: '800px',
            background: '#0a0e27',
            color: '#f0f9ff',
            showCloseButton: true,
            showConfirmButton: false,
            customClass: {
                popup: 'swal-custom-popup',
                title: 'swal-custom-title',
                htmlContainer: 'swal-custom-text'
            }
        });
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error', 'Error al generar el análisis de outliers', 'error');
    });
}

// Exportar funciones para uso global
window.renameColumn = renameColumn;
window.deleteColumn = deleteColumn;
window.showUniqueValues = showUniqueValues;
window.showOutliers = showOutliers;