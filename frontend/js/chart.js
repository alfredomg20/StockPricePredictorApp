import { PRIMARY_COLOR } from './config.js';

export class predictionChart {
    static chartInstance = null;

    static renderChart(predictions) {
        // Format data for chart - parse dates as local to avoid timezone offset issues
        const dates = predictions.map(p => {
            const dateParts = p.predicted_date.split('-');
            return new Date(parseInt(dateParts[0]), parseInt(dateParts[1]) - 1, parseInt(dateParts[2]));
        });
        const prices = predictions.map(p => p.predicted_price);

        // Get the container element
        const chartContainer = document.getElementById('predictionChart');
        if (!chartContainer) {
            console.error('Chart container not found');
            return;
        }
        
        // Ensure we have a canvas element
        let chartCanvas = chartContainer.querySelector('canvas');
        if (!chartCanvas) {
            chartCanvas = document.createElement('canvas');
            chartContainer.appendChild(chartCanvas);
            
            // Set canvas dimensions to match container
            chartCanvas.style.width = '100%';
            chartCanvas.style.height = '100%';
        }
        
        // Destroy previous chart if it exists
        if (this.chartInstance) {
            this.chartInstance.destroy();
        }
        
        // Create the new chart
        this.chartInstance = new Chart(chartCanvas, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Predicted Price',
                    data: prices,
                    backgroundColor: PRIMARY_COLOR,
                    borderColor: PRIMARY_COLOR,
                    borderWidth: 2,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    pointBackgroundColor: PRIMARY_COLOR,
                    pointBorderColor: '#1a1a1a',
                    pointBorderWidth: 2,
                    tension: 0.1 // slight curve to the line
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // allow the chart to fill the container
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 26, 26, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: PRIMARY_COLOR,
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return `$${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM dd'
                            },
                            tooltipFormat: 'MMM dd, yyyy'
                        },
                        ticks: {
                            color: '#ffffff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (USD)',
                            color: '#ffffff'
                        },
                        ticks: {
                            color: '#ffffff',
                            callback: function(value) {
                                return '$' + value.toFixed(2);
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }
}