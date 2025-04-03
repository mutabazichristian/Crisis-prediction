// Dashboard functionality
document.addEventListener('DOMContentLoaded', function() {
    // Load dashboard stats
    loadDashboardStats();
    
    // Load recent predictions if available
    loadRecentPredictions();
    
    // Setup country filter change listener
    const countryFilter = document.getElementById('countryFilter');
    if (countryFilter) {
        countryFilter.addEventListener('change', function() {
            loadDashboardStats(this.value);
        });
    }
});

/**
 * Load dashboard statistics
 */
async function loadDashboardStats(countryId = 'all') {
    try {
        // In a real app, you would fetch this from your backend API
        // For demo purposes, we'll use sample data
        const stats = getSampleDashboardStats(countryId);
        updateDashboardUI(stats);
    } catch (error) {
        console.error('Error loading dashboard stats:', error);
        displayDashboardError('Failed to load dashboard statistics');
    }
}

/**
 * Get sample dashboard statistics
 */
function getSampleDashboardStats(countryId) {
    // Sample data for dashboard
    const allCountriesStats = {
        totalPredictions: 1245,
        crisisDetected: 187,
        averageProbability: 0.31,
        modelAccuracy: 0.89,
        recentTrend: 'decreasing',
        highRiskCountries: ['Zimbabwe', 'Nigeria', 'Central African Republic'],
        predictionDistribution: {
            lowRisk: 58,
            mediumRisk: 29,
            highRisk: 13
        }
    };
    
    // Country-specific stats (simplified for demo)
    const countryStats = {
        '0': { // Algeria
            totalPredictions: 95,
            crisisDetected: 12,
            averageProbability: 0.24,
            modelAccuracy: 0.91,
            recentTrend: 'stable',
            predictionDistribution: {
                lowRisk: 68,
                mediumRisk: 23,
                highRisk: 9
            }
        },
        '8': { // Nigeria
            totalPredictions: 120,
            crisisDetected: 36,
            averageProbability: 0.42,
            modelAccuracy: 0.87,
            recentTrend: 'increasing',
            predictionDistribution: {
                lowRisk: 42,
                mediumRisk: 33,
                highRisk: 25
            }
        },
        '9': { // South Africa
            totalPredictions: 110,
            crisisDetected: 22,
            averageProbability: 0.29,
            modelAccuracy: 0.92,
            recentTrend: 'decreasing',
            predictionDistribution: {
                lowRisk: 61,
                mediumRisk: 28,
                highRisk: 11
            }
        }
    };
    
    return countryId === 'all' ? allCountriesStats : (countryStats[countryId] || allCountriesStats);
}

/**
 * Update dashboard UI with statistics
 */
function updateDashboardUI(stats) {
    // Update summary stats
    updateElement('totalPredictions', stats.totalPredictions);
    updateElement('crisisDetected', stats.crisisDetected);
    updateElement('averageProbability', (stats.averageProbability * 100).toFixed(1) + '%');
    updateElement('modelAccuracy', (stats.modelAccuracy * 100).toFixed(1) + '%');
    
    // Update trend indicator
    const trendElement = document.getElementById('recentTrend');
    if (trendElement) {
        let trendHTML = '';
        if (stats.recentTrend === 'increasing') {
            trendHTML = '<i class="fas fa-arrow-up text-danger"></i> Increasing Risk';
        } else if (stats.recentTrend === 'decreasing') {
            trendHTML = '<i class="fas fa-arrow-down text-success"></i> Decreasing Risk';
        } else {
            trendHTML = '<i class="fas fa-arrows-alt-h text-warning"></i> Stable Risk';
        }
        trendElement.innerHTML = trendHTML;
    }
    
    // Update high risk countries list if available
    if (stats.highRiskCountries) {
        const highRiskElement = document.getElementById('highRiskCountries');
        if (highRiskElement) {
            highRiskElement.innerHTML = stats.highRiskCountries.map(country => 
                `<span class="badge bg-danger me-1">${country}</span>`
            ).join('');
        }
    }
    
    // Update distribution chart if available
    if (stats.predictionDistribution) {
        updateDistributionChart(stats.predictionDistribution);
    }
}

/**
 * Update prediction distribution chart
 */
function updateDistributionChart(distribution) {
    const chartElement = document.getElementById('distributionChart');
    if (!chartElement) return;
    
    const ctx = chartElement.getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.distributionChart) {
        window.distributionChart.destroy();
    }
    
    // Create new chart
    window.distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [distribution.lowRisk, distribution.mediumRisk, distribution.highRisk],
                backgroundColor: [
                    '#198754', // Green - Low Risk
                    '#ffc107', // Yellow - Medium Risk
                    '#dc3545'  // Red - High Risk
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((acc, val) => acc + val, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${percentage}% (${value})`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Load recent predictions for dashboard
 */
async function loadRecentPredictions() {
    try {
        // In a real app, you would fetch this from your backend API
        // For demo purposes, we'll use sample data
        const predictions = getSampleRecentPredictions();
        
        // Update UI with recent predictions
        const container = document.getElementById('recentPredictions');
        if (!container) return;
        
        if (predictions.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent predictions available</p>';
            return;
        }
        
        const predictionHTML = predictions.map(pred => {
            let statusClass = 'bg-success';
            let riskText = 'Low Risk';
            
            if (pred.probability > 0.7) {
                statusClass = 'bg-danger';
                riskText = 'High Risk';
            } else if (pred.probability > 0.3) {
                statusClass = 'bg-warning';
                riskText = 'Medium Risk';
            }
            
            return `
                <div class="d-flex justify-content-between align-items-center mb-3 p-3 bg-light rounded">
                    <div>
                        <h6 class="mb-0">${pred.country}</h6>
                        <small class="text-muted">${pred.date}</small>
                    </div>
                    <div class="text-end">
                        <span class="badge ${statusClass}">${riskText}</span>
                        <div>${(pred.probability * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = predictionHTML;
    } catch (error) {
        console.error('Error loading recent predictions:', error);
        const container = document.getElementById('recentPredictions');
        if (container) {
            container.innerHTML = '<div class="alert alert-warning">Failed to load recent predictions</div>';
        }
    }
}

/**
 * Get sample recent predictions
 */
function getSampleRecentPredictions() {
    return [
        {
            country: 'Nigeria',
            date: '2025-03-28',
            probability: 0.76,
            features: {
                inflation: 12.3,
                exchange_rate: 415.2,
                gdp_growth: -1.8
            }
        },
        {
            country: 'Kenya',
            date: '2025-03-25',
            probability: 0.42,
            features: {
                inflation: 8.1,
                exchange_rate: 108.5,
                gdp_growth: 0.9
            }
        },
        {
            country: 'South Africa',
            date: '2025-03-22',
            probability: 0.28,
            features: {
                inflation: 6.2,
                exchange_rate: 15.1,
                gdp_growth: 1.2
            }
        },
        {
            country: 'Egypt',
            date: '2025-03-20',
            probability: 0.63,
            features: {
                inflation: 14.6,
                exchange_rate: 15.7,
                gdp_growth: -0.5
            }
        },
        {
            country: 'Morocco',
            date: '2025-03-18',
            probability: 0.19,
            features: {
                inflation: 4.8,
                exchange_rate: 10.2,
                gdp_growth: 2.3
            }
        }
    ];
}

/**
 * Helper function to update element text content
 */
function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

/**
 * Display error message on dashboard
 */
function displayDashboardError(message) {
    const container = document.querySelector('.container');
    if (!container) return;
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.innerHTML = `
        <strong><i class="fas fa-exclamation-circle me-2"></i> Error:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    container.prepend(errorDiv);
}
