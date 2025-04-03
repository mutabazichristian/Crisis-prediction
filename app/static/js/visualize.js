// Fetch historical data for visualization
document.addEventListener('DOMContentLoaded', function() {
    // Fetch data
    fetchVisualizationData()
        .then(data => {
            renderCharts(data);
        })
        .catch(error => {
            console.error('Error loading visualization data:', error);
            displayError('Failed to load visualization data. Please try again later.');
        });
});

/**
 * Fetch visualization data from API
 */
async function fetchVisualizationData() {
    try {
        // In a real app, you would fetch this from your backend API
        // For demo purposes, we'll use sample data
        return getSampleData();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Sample data for visualization
 */
function getSampleData() {
    return {
        defaultRates: {
            labels: ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
            datasets: [
                {
                    label: 'Nigeria',
                    data: [2.1, 2.3, 3.1, 3.4, 4.2, 5.1, 4.8, 4.2, 3.9, 4.1, 5.3, 4.9, 4.5, 4.1],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Kenya',
                    data: [1.8, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 3.2, 2.9, 3.1, 4.2, 3.8, 3.5, 3.2],
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'South Africa',
                    data: [3.2, 3.6, 4.1, 4.5, 5.2, 5.8, 5.4, 5.0, 4.8, 5.1, 6.2, 5.7, 5.3, 5.0],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        loanDefaultRates: {
            labels: ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
            datasets: [
                {
                    label: 'Nigeria',
                    data: [5.2, 5.4, 6.2, 6.8, 7.5, 8.3, 7.9, 7.5, 7.1, 7.6, 9.2, 8.4, 7.8, 7.2],
                    borderColor: '#0d6efd',
                    backgroundColor: '#0d6efd',
                }
                ,
                {
                    label: 'Kenya',
                    data: [4.1, 4.3, 4.8, 5.2, 5.5, 6.1, 6.3, 5.9, 5.5, 5.8, 7.6, 6.9, 6.4, 5.9],
                    borderColor: '#198754',
                    backgroundColor: '#198754',
                },
                {
                    label: 'South Africa',
                    data: [6.8, 7.3, 8.1, 8.9, 9.7, 10.5, 9.8, 9.4, 9.0, 9.5, 11.8, 10.5, 9.7, 9.1],
                    borderColor: '#dc3545',
                    backgroundColor: '#dc3545',
                }
            ]
        },
        crisisOccurrences: {
            countries: ['Algeria', 'Angola', 'CAR', 'Egypt', 'Ivory Coast', 'Kenya', 'Mauritius', 'Morocco', 'Nigeria', 'South Africa', 'Tunisia', 'Zambia', 'Zimbabwe'],
            occurrences: [1, 2, 3, 2, 4, 1, 0, 1, 3, 2, 1, 2, 5]
        },
        riskFactors: {
            labels: ['Exchange Rate Volatility', 'Inflation', 'GDP Decline', 'External Debt', 'Banking Sector Weakness'],
            datasets: [
                {
                    label: 'Impact Factor',
                    data: [0.78, 0.85, 0.71, 0.68, 0.91],
                    backgroundColor: [
                        'rgba(13, 110, 253, 0.7)',
                        'rgba(25, 135, 84, 0.7)',
                        'rgba(255, 193, 7, 0.7)',
                        'rgba(220, 53, 69, 0.7)',
                        'rgba(13, 202, 240, 0.7)'
                    ],
                    borderWidth: 1
                }
            ]
        }
    };
}

/**
 * Render visualizations with Chart.js
 */
function renderCharts(data) {
    // Default Rate Chart
    const defaultRateCtx = document.getElementById('defaultRateChart').getContext('2d');
    new Chart(defaultRateCtx, {
        type: 'line',
        data: data.defaultRates,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Default Rates Over Time (%)',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Default Rate (%)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Year'
                    }
                }
            }
        }
    });

    // Loan Default Rate Chart
    const loanDefaultRateCtx = document.getElementById('loanDefaultRateChart').getContext('2d');
    new Chart(loanDefaultRateCtx, {
        type: 'bar',
        data: data.loanDefaultRates,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Loan Default Rates Over Time (%)',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Loan Default Rate (%)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Year'
                    }
                }
            }
        }
    });
}

/**
 * Display error message on the page
 */
function displayError(message) {
    const container = document.querySelector('.container');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger mt-4';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i> ${message}`;
    container.prepend(errorDiv);
}
