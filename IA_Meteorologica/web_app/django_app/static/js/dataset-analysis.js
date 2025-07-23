// Dataset Analysis Advanced Functions

// Variable para almacenar análisis generales del dataset
let generalAnalysisData = {};

// Función para generar análisis avanzado de variable
async function generateAdvancedAnalysis(analysisId, analysisType) {
    const select = document.getElementById(`variableSelect${analysisId}`);
    const selectedVariable = select.value;
    
    if (!selectedVariable) {
        showNotification('Erreur', 'Veuillez sélectionner une variable', 'warning');
        return;
    }
    
    const currentDataset = window.selectedDataset;
    if (!currentDataset || !currentDataset.id) {
        showNotification('Erreur', 'Aucun dataset sélectionné', 'error');
        return;
    }
    
    try {
        let url = `/api/datasets/${currentDataset.id}/columns/${encodeURIComponent(selectedVariable)}/analysis/?type=${analysisType}`;
        
        // Para scatter plot necesitamos una segunda variable
        if (analysisType === 'scatter') {
            const secondVariable = prompt('Sélectionnez la deuxième variable pour le scatter plot:');
            if (!secondVariable) return;
            url += `&second_column=${encodeURIComponent(secondVariable)}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Erreur lors de la génération de l\'analyse');
        }
        
        // Mostrar el análisis
        displayAdvancedAnalysis(analysisId, data);
        
    } catch (error) {
        showNotification('Erreur', error.message, 'error');
    }
}

// Función para mostrar análisis avanzado
function displayAdvancedAnalysis(analysisId, data) {
    const container = document.getElementById(`analysisResultContainer${analysisId}`);
    if (!container) {
        // Crear contenedor si no existe
        const statsContainer = document.getElementById(`variableStats${analysisId}`);
        const newContainer = document.createElement('div');
        newContainer.id = `analysisResultContainer${analysisId}`;
        newContainer.className = 'mt-4';
        statsContainer.parentElement.insertBefore(newContainer, statsContainer.nextSibling);
    }
    
    let content = `
        <div class="card bg-dark border-secondary">
            <div class="card-header">
                <h6 class="mb-0 text-info">
                    <i class="bi bi-graph-up-arrow"></i> 
                    ${getAnalysisTitle(data.analysis_type)} - ${data.column || data.columns?.join(' vs ')}
                </h6>
            </div>
            <div class="card-body">
                <img src="${data.image}" class="img-fluid" alt="${data.analysis_type}">
    `;
    
    // Agregar estadísticas específicas según el tipo
    if (data.statistics) {
        content += '<div class="mt-3">';
        
        switch(data.analysis_type) {
            case 'outlier_map':
                content += `
                    <div class="alert alert-info">
                        <strong>Outliers détectés:</strong> ${data.statistics.outlier_count} 
                        (${data.statistics.outlier_percentage.toFixed(2)}%)
                        <br>
                        <small>IQR: ${data.statistics.iqr.toFixed(2)}, 
                        Limites: [${data.statistics.lower_bound.toFixed(2)}, ${data.statistics.upper_bound.toFixed(2)}]</small>
                    </div>
                `;
                break;
                
            case 'boxplot':
                content += `
                    <div class="row text-center">
                        <div class="col">
                            <small class="text-muted">Min</small><br>
                            <strong>${data.statistics.min.toFixed(2)}</strong>
                        </div>
                        <div class="col">
                            <small class="text-muted">Q1</small><br>
                            <strong>${data.statistics.q1.toFixed(2)}</strong>
                        </div>
                        <div class="col">
                            <small class="text-muted">Médiane</small><br>
                            <strong>${data.statistics.median.toFixed(2)}</strong>
                        </div>
                        <div class="col">
                            <small class="text-muted">Moyenne</small><br>
                            <strong class="text-success">${data.statistics.mean.toFixed(2)}</strong>
                        </div>
                        <div class="col">
                            <small class="text-muted">Q3</small><br>
                            <strong>${data.statistics.q3.toFixed(2)}</strong>
                        </div>
                        <div class="col">
                            <small class="text-muted">Max</small><br>
                            <strong>${data.statistics.max.toFixed(2)}</strong>
                        </div>
                    </div>
                `;
                break;
                
            case 'scatter':
                content += `
                    <div class="alert alert-info">
                        <strong>Corrélation:</strong> ${data.statistics.correlation.toFixed(3)}
                        <br>
                        <strong>Équation de régression:</strong> 
                        y = ${data.statistics.regression_slope.toFixed(3)}x + ${data.statistics.regression_intercept.toFixed(3)}
                        <br>
                        <small>Points de données: ${data.statistics.data_points}</small>
                    </div>
                `;
                break;
        }
        
        content += '</div>';
    }
    
    content += `
            </div>
        </div>
    `;
    
    document.getElementById(`analysisResultContainer${analysisId}`).innerHTML = content;
}

// Función para generar análisis general del dataset
async function generateGeneralAnalysis(analysisType) {
    const currentDataset = window.selectedDataset;
    if (!currentDataset || !currentDataset.id) {
        showNotification('Erreur', 'Aucun dataset sélectionné', 'error');
        return;
    }
    
    try {
        let url = `/api/datasets/${currentDataset.id}/analysis/?type=${analysisType}`;
        
        // Para LASSO necesitamos una variable objetivo
        if (analysisType === 'lasso') {
            const targetVariable = prompt('Sélectionnez la variable cible (target) pour LASSO:');
            if (!targetVariable) return;
            url += `&target=${encodeURIComponent(targetVariable)}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Erreur lors de la génération de l\'analyse');
        }
        
        // Mostrar el análisis
        displayGeneralAnalysis(data);
        
        // Guardar datos
        generalAnalysisData[analysisType] = data;
        
    } catch (error) {
        showNotification('Erreur', error.message, 'error');
    }
}

// Función para mostrar análisis general
function displayGeneralAnalysis(data) {
    const container = document.getElementById('generalAnalysisContainer');
    if (!container) {
        // Crear contenedor si no existe
        const analysisSection = document.querySelector('#analysisContainer').parentElement;
        const newSection = document.createElement('div');
        newSection.className = 'row mb-4';
        newSection.innerHTML = `
            <div class="col-md-12">
                <h5 class="text-primary mb-3"><i class="bi bi-diagram-3"></i> Analyses Générales du Dataset</h5>
                <div id="generalAnalysisContainer"></div>
            </div>
        `;
        analysisSection.appendChild(newSection);
    }
    
    let content = `
        <div class="card bg-dark border-primary mb-3">
            <div class="card-header">
                <h6 class="mb-0 text-info">
                    <i class="bi bi-graph-up-arrow"></i> 
                    ${getAnalysisTitle(data.analysis_type)}
                </h6>
            </div>
            <div class="card-body">
                <img src="${data.image}" class="img-fluid" alt="${data.analysis_type}">
    `;
    
    // Agregar información específica según el tipo
    if (data.statistics) {
        content += '<div class="mt-3">';
        
        switch(data.analysis_type) {
            case 'correlation_matrix':
                content += `
                    <h6 class="text-info">Corrélations Fortes (|r| > 0.5):</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-hover">
                            <thead>
                                <tr>
                                    <th>Variable 1</th>
                                    <th>Variable 2</th>
                                    <th>Corrélation</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                data.statistics.strong_correlations.forEach(corr => {
                    const color = corr.correlation > 0 ? 'text-success' : 'text-danger';
                    content += `
                        <tr>
                            <td>${corr.var1}</td>
                            <td>${corr.var2}</td>
                            <td class="${color}">${corr.correlation.toFixed(3)}</td>
                        </tr>
                    `;
                });
                
                content += `
                            </tbody>
                        </table>
                    </div>
                `;
                break;
                
            case 'pca':
                content += `
                    <div class="alert alert-info">
                        <strong>Composantes nécessaires pour 95% de variance:</strong> 
                        ${data.statistics.n_components_95_variance} sur ${data.statistics.total_variables}
                    </div>
                    <h6 class="text-info">Contribution des Variables aux Composantes Principales:</h6>
                `;
                
                Object.entries(data.statistics.principal_components).forEach(([pc, info]) => {
                    content += `
                        <div class="mb-3">
                            <strong>${pc}</strong> (${(info.variance_explained * 100).toFixed(1)}% de variance)
                            <ul class="small">
                    `;
                    info.top_contributors.forEach(([var_name, loading]) => {
                        content += `<li>${var_name}: ${loading.toFixed(3)}</li>`;
                    });
                    content += `
                            </ul>
                        </div>
                    `;
                });
                break;
                
            case 'lasso':
                content += `
                    <div class="alert alert-info">
                        <strong>Meilleur Alpha:</strong> ${data.statistics.best_alpha.toExponential(2)}
                        <br>
                        <strong>Score R²:</strong> ${data.statistics.best_score.toFixed(3)}
                        <br>
                        <strong>Variables sélectionnées:</strong> ${data.statistics.n_features_selected} sur ${data.statistics.total_features}
                    </div>
                    
                    <h6 class="text-info">Variables Importantes:</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-hover">
                            <thead>
                                <tr>
                                    <th>Variable</th>
                                    <th>Coefficient</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                data.statistics.selected_features.forEach(([feature, coef]) => {
                    const color = coef > 0 ? 'text-success' : 'text-danger';
                    content += `
                        <tr>
                            <td>${feature}</td>
                            <td class="${color}">${coef.toFixed(4)}</td>
                        </tr>
                    `;
                });
                
                content += `
                            </tbody>
                        </table>
                    </div>
                `;
                break;
        }
        
        content += '</div>';
    }
    
    content += `
            </div>
        </div>
    `;
    
    // Agregar al contenedor
    const existingCard = document.querySelector(`[data-analysis-type="${data.analysis_type}"]`);
    if (existingCard) {
        existingCard.remove();
    }
    
    const newCard = document.createElement('div');
    newCard.setAttribute('data-analysis-type', data.analysis_type);
    newCard.innerHTML = content;
    
    document.getElementById('generalAnalysisContainer').appendChild(newCard);
}

// Función auxiliar para obtener título del análisis
function getAnalysisTitle(analysisType) {
    const titles = {
        'histogram': 'Histogramme',
        'outlier_map': 'Carte des Outliers',
        'boxplot': 'Box Plot',
        'scatter': 'Scatter Plot',
        'correlation_matrix': 'Matrice de Corrélation',
        'pca': 'Analyse en Composantes Principales (PCA)',
        'lasso': 'Sélection de Variables LASSO'
    };
    return titles[analysisType] || analysisType;
}

// Función para actualizar los botones de análisis
function updateAnalysisButtons(analysisId) {
    const select = document.getElementById(`variableSelect${analysisId}`);
    if (!select || !select.value) {
        // Si no hay variable seleccionada, solo mostrar el botón de histograma
        const buttonContainer = document.getElementById(`analysisButtons${analysisId}`);
        if (buttonContainer) {
            buttonContainer.innerHTML = `
                <button class="btn btn-primary" onclick="generateHistogram(${analysisId})">
                    <i class="bi bi-bar-chart"></i> Histogramme
                </button>
            `;
        }
        return;
    }
    
    const selectedOption = select.options[select.selectedIndex];
    const isNumeric = selectedOption.getAttribute('data-numeric') === 'true';
    
    const buttonContainer = document.getElementById(`analysisButtons${analysisId}`);
    if (!buttonContainer) return;
    
    let buttons = `
        <button class="btn btn-primary me-2 mb-2" onclick="generateHistogram(${analysisId})">
            <i class="bi bi-bar-chart"></i> Histogramme
        </button>
    `;
    
    if (isNumeric) {
        buttons += `
            <button class="btn btn-warning me-2 mb-2" onclick="generateAdvancedAnalysis(${analysisId}, 'outlier_map')">
                <i class="bi bi-exclamation-triangle"></i> Outliers
            </button>
            <button class="btn btn-info me-2 mb-2" onclick="generateAdvancedAnalysis(${analysisId}, 'boxplot')">
                <i class="bi bi-box"></i> Box Plot
            </button>
            <button class="btn btn-success me-2 mb-2" onclick="generateAdvancedAnalysis(${analysisId}, 'scatter')">
                <i class="bi bi-scatter"></i> Scatter
            </button>
        `;
    }
    
    buttonContainer.innerHTML = buttons;
}

// Exportar funciones para uso global
window.generateAdvancedAnalysis = generateAdvancedAnalysis;
window.generateGeneralAnalysis = generateGeneralAnalysis;
window.updateAnalysisButtons = updateAnalysisButtons;