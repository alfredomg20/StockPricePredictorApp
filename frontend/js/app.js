import { pricePredictorAPI } from './api.js';
import { predictionChart } from './chart.js';
import { predictionTable } from './table.js';
import { cache } from './cache.js';

/**
 * Main App class for the stock price prediction application
 */
class StockPredictorApp {
  constructor() {
    // Application state
    this.currentPredictions = null;
    this.currentTicker = null;
    this.trainedModel = {
      ticker: null,
      forecastDays: null
    };
    
    // Cache TTL constants (1 hour = 3600000 ms)
    this.CACHE_TTL = {
      METRICS: 3600000,
      PREDICTIONS: 3600000
    };
    
    // DOM element references
    this.elements = {
      stockSelect: null,
      daysInput: null,
      predictBtn: null,
      predictionForm: null,
      metricsCard: null,
      metricsContainer: null,
      showPredictionsBtn: null,
      predictionCard: null,
      predictionFigure: null,
      errorAlert: null,
      tableDisplayRadio: null,
      chartDisplayRadio: null
    };
  }

  /**
   * Initializes the application after the DOM is loaded
   */
  init() {
    try {
      this.cacheElements();
      this.populateTickerDropdown();
      this.bindEvents();
      this.loadCachedData();
    } catch (error) {
      this.showError('Could not initialize the application. Please try again later.');
      console.error('Initialization error:', error);
    }
  }

  /**
   * Caches references to DOM elements
   */
  cacheElements() {
    this.elements = {
      stockSelect: document.getElementById('stockSelect'),
      daysInput: document.getElementById('daysToPredict'),
      predictBtn: document.getElementById('predictBtn'),
      predictionForm: document.getElementById('predictionForm'),
      metricsCard: document.getElementById('metricsCard'),
      metricsContainer: document.getElementById('metricsContainer'),
      showPredictionsBtn: document.getElementById('showPredictionsBtn'),
      predictionCard: document.getElementById('predictionCard'),
      predictionFigure: document.getElementById('predictionFigure'),
      errorAlert: document.getElementById('errorAlert'),
      tableDisplayRadio: document.getElementById('tableDisplayRadio'),
      chartDisplayRadio: document.getElementById('chartDisplayRadio')
    };
  }

  /**
   * Binds events to DOM elements
   */
  bindEvents() {
    const { predictionForm, showPredictionsBtn, tableDisplayRadio, chartDisplayRadio } = this.elements;
    
    if (predictionForm) {
      predictionForm.addEventListener('submit', this.handleFormSubmit.bind(this));
    }
    
    if (showPredictionsBtn) {
      showPredictionsBtn.addEventListener('click', this.showPredictions.bind(this));
    }
    
    if (tableDisplayRadio && chartDisplayRadio) {
      tableDisplayRadio.addEventListener('change', this.togglePredictionDisplay.bind(this));
      chartDisplayRadio.addEventListener('change', this.togglePredictionDisplay.bind(this));
    }
  }

  /**
   * Populates the ticker dropdown with available values
   */
  populateTickerDropdown() {
    const { stockSelect } = this.elements;
    const tickers = pricePredictorAPI.getAvailableTickers();
    
    tickers.forEach(ticker => {
      const option = document.createElement('option');
      option.value = ticker;
      option.textContent = ticker;
      stockSelect.appendChild(option);
    });
  }

  /**
   * Loads cached user data on app initialization
   */
  loadCachedData() {
    // Load last selected ticker
    const lastTicker = cache.get('lastSelectedTicker');
    if (lastTicker && this.elements.stockSelect) {
      this.elements.stockSelect.value = lastTicker;
    }

    // Load last forecast days
    const lastForecastDays = cache.get('lastForecastDays');
    if (lastForecastDays && this.elements.daysInput) {
      this.elements.daysInput.value = lastForecastDays;
    }

    // Load trained model info
    const trainedModel = cache.get('trainedModel');
    if (trainedModel) {
      this.trainedModel = trainedModel;
      
      // Check if we have cached metrics for this model
      const metricsKey = cache.getMetricsKey(trainedModel.ticker, trainedModel.forecastDays);
      const cachedMetrics = cache.get(metricsKey);
      if (cachedMetrics) {
        this.showMetrics(null, cachedMetrics);
      }
    }
  }

  /**
   * Handles the prediction form submission
   * @param {Event} e - Form submit event
   */
  async handleFormSubmit(e) {
    e.preventDefault();
    
    // Reset UI
    this.hideError();
    
    // Get form values
    const { stockSelect, daysInput, predictBtn, metricsCard, metricsContainer } = this.elements;
    const ticker = stockSelect.value;
    const forecastDays = parseInt(daysInput.value);
    
    // Cache user inputs
    cache.set('lastSelectedTicker', ticker);
    cache.set('lastForecastDays', forecastDays);
    
    try {
      // Disable form while processing
      if (predictBtn) {
        this.setButtonLoading(predictBtn, 'Processing...');
      }
      
      // Show loading indicator in metrics card
      if (metricsCard && !metricsCard.classList.contains('d-none') && metricsContainer) {
        this.showLoadingInElement(metricsContainer, `Training model for ${ticker}...`);
        this.updateCardTitle(metricsCard, `Training model for ${ticker} (${forecastDays} days)`);
      }
      
      // Check cache first for metrics
      const metricsKey = cache.getMetricsKey(ticker, forecastDays);
      const cachedMetrics = cache.get(metricsKey);
      
      if (cachedMetrics) {
        // Use cached metrics
        this.trainedModel = { ticker, forecastDays };
        cache.set('trainedModel', this.trainedModel);
        this.showMetrics(null, cachedMetrics);
        return;
      }
      
      // Start training or check if model exists
      const trainingResponse = await pricePredictorAPI.startTraining(ticker, forecastDays);
      
      await this.handleTrainingResponse(trainingResponse, ticker, forecastDays);
      
    } catch (error) {
      console.error('Error on prediction form submit:', error);
      this.showError(`Error starting the prediction process: ${error.message}`);
    } finally {
      if (predictBtn) {
        this.resetButton(predictBtn, 'Train Model');
      }
    }
  }

  /**
   * Handles the training process response
   * @param {Object} trainingResponse - Training API response
   * @param {string} ticker - Stock symbol
   * @param {number} forecastDays - Days to predict
   */
  async handleTrainingResponse(trainingResponse, ticker, forecastDays) {
    // Save trained model configuration
    this.trainedModel = {
      ticker: ticker,
      forecastDays: forecastDays
    };
    
    // Cache trained model info (no TTL)
    cache.set('trainedModel', this.trainedModel);

    if (trainingResponse.status === 'skipped' || trainingResponse.status === 'completed') {
      const result = {
        ticker,
        forecast_days: forecastDays,
        metrics: trainingResponse.metrics
      };
      
      // Cache metrics with TTL
      if (trainingResponse.metrics) {
        const metricsKey = cache.getMetricsKey(ticker, forecastDays);
        cache.set(metricsKey, result, this.CACHE_TTL.METRICS);
      }
      
      this.showMetrics('The model is ready for predictions!', result);
    } else if (trainingResponse.task_id) {
      try {
        const result = await pricePredictorAPI.getTrainingResult(trainingResponse.task_id);
        
        // Cache metrics with TTL
        if (result.metrics) {
          const metricsKey = cache.getMetricsKey(ticker, forecastDays);
          cache.set(metricsKey, result, this.CACHE_TTL.METRICS);
        }
        
        this.showMetrics('Training completed successfully!', result);
      } catch (error) {
        console.error('Error getting training result:', error);
        this.showMetrics('The model is ready for predictions.', {
          ticker,
          forecast_days: forecastDays
        });
      }
    } else {
      this.showMetrics('The model is ready for predictions.', {
        ticker,
        forecast_days: forecastDays
      });
    }
  }

  /**
   * Displays model metrics in the UI
   * @param {string} message - Message to show to the user (null to skip alert)
   * @param {Object} result - Training result
   */
  showMetrics(message, result) {
    const { metricsCard, metricsContainer, showPredictionsBtn, predictionCard } = this.elements;
    
    if (!metricsCard) {
      console.error('metricsCard element not found');
      return;
    }
    
    // Show card if it was hidden
    if (metricsCard.classList.contains('d-none')) {
      metricsCard.classList.remove('d-none');
    }
    
    // Update card title
    this.updateCardTitle(metricsCard, `${result.ticker} Model Performance (${result.forecast_days} days)`);
    
    // Show alert only if message is provided
    if (message) {
      this.showTemporaryAlert(message);
    }
    
    // Update metrics if we have new ones
    if (result.metrics && metricsContainer) {
      this.renderMetricsCards(metricsContainer, result.metrics);
    }
    
    // Ensure the predictions button is visible
    if (showPredictionsBtn) {
      showPredictionsBtn.classList.remove('d-none');
    }
  }

  /**
   * Renders metric cards in the specified container
   * @param {HTMLElement} container - Container for metrics
   * @param {Object} metrics - Metrics data to display
   */
  renderMetricsCards(container, metrics) {
    // Clear previous content
    container.innerHTML = '';
    
    // Create row for metrics
    const metricsRow = document.createElement('div');
    metricsRow.className = 'row g-2';
    container.appendChild(metricsRow);
    
    // Define metrics to display
    const metricsConfig = [
      { label: 'Mean Absolute Error', value: '$' + metrics.mae.toFixed(2), key: 'mae' },
      { label: 'R-squared', value: metrics.r2.toFixed(4), key: 'r2' },
      { label: 'Mean Absolute Percentage Error', value: `${(metrics.mape * 100).toFixed(2)}%`, key: 'mape' },
      { label: 'Max Error', value: '$' + metrics.max_ae.toFixed(2), key: 'max_ae' },
      { label: 'Training Samples', value: metrics.train_samples, key: 'train_samples' },
      { label: 'Test Samples', value: metrics.test_samples, key: 'test_samples' }
    ];
    
    // Create metric cards
    metricsConfig.forEach(metric => {
      const metricEl = document.createElement('div');
      metricEl.className = 'col-sm-6';
      metricEl.innerHTML = `
        <div class="metric-card text-light">
          <div class="metric-value">${metric.value}</div>
          <div class="metric-label">${metric.label}</div>
        </div>
      `;
      metricsRow.appendChild(metricEl);
    });
  }

  /**
   * Shows a temporary alert
   * @param {string} message - Message to display
   */
  showTemporaryAlert(message) {
    // First, remove any existing alert
    const existingAlertContainer = document.querySelector('.metrics-alert-container');
    if (existingAlertContainer) {
      existingAlertContainer.remove();
    }
    
    // Create container for alert
    const alertContainer = document.createElement('div');
    alertContainer.className = 'metrics-alert-container';
    
    // Create Bootstrap alert
    const alertEl = document.createElement('div');
    alertEl.className = 'alert alert-success metrics-alert shadow';
    alertEl.setAttribute('role', 'alert');
    alertEl.textContent = message;
    
    alertContainer.appendChild(alertEl);
    document.body.appendChild(alertContainer);
    
    // Auto-hide the alert after 5 seconds
    setTimeout(() => {
      alertEl.classList.add('fade-out');
      setTimeout(() => {
        if (document.body.contains(alertContainer)) {
          alertContainer.remove();
        }
      }, 500); // Match transition time
    }, 5000);
  }

  /**
   * Shows predictions based on the trained model
   */
  async showPredictions() {
    const { stockSelect, daysInput, showPredictionsBtn, predictionFigure } = this.elements;
    
    try {
      // Use ticker from form but forecast days from trained model
      const ticker = stockSelect.value;
      
      // Check that a trained model exists
      if (!this.trainedModel.ticker || !this.trainedModel.forecastDays) {
        throw new Error('Please train a model first before making predictions');
      }
      
      // Check that the ticker matches the trained model
      if (ticker !== this.trainedModel.ticker) {
        throw new Error(`The current model is trained for ${this.trainedModel.ticker}. Please train a model for ${ticker} first`);
      }
      
      const forecastDays = this.trainedModel.forecastDays;
      
      // Show notice if form value does not match trained model
      if (parseInt(daysInput.value) !== forecastDays) {
        this.showTemporaryAlert(`Using ${forecastDays} days from the trained model instead of ${daysInput.value} days from the form`);
      }
      
      // Check cache first for predictions
      const predictionsKey = cache.getPredictionsKey(ticker, forecastDays);
      const cachedPredictions = cache.get(predictionsKey);
      
      if (cachedPredictions) {
        // Use cached predictions
        this.currentPredictions = cachedPredictions;
        this.currentTicker = ticker;
        this.preparePredictionUI(ticker, forecastDays);
        this.renderPredictions();
        return;
      }
      
      // Show loading state
      if (showPredictionsBtn) {
        this.setButtonLoading(showPredictionsBtn, 'Loading...');
      }
      
      // Prepare UI to show predictions
      this.preparePredictionUI(ticker, forecastDays);
      
      // Make prediction
      const predictionResponse = await pricePredictorAPI.makePrediction(ticker, forecastDays);

      if (predictionResponse.success && predictionFigure) {
        // Store current predictions
        this.currentPredictions = predictionResponse.predictions;
        this.currentTicker = ticker;
        
        // Cache predictions with TTL
        cache.set(predictionsKey, predictionResponse.predictions, this.CACHE_TTL.PREDICTIONS);
        
        // Show predictions
        this.renderPredictions();
        
        // Reset button
        if (showPredictionsBtn) {
          this.resetButton(showPredictionsBtn, 'Predict');
        }
        
        // Show success message
        this.showTemporaryAlert(`Successfully predicted prices for ${ticker} for the next ${forecastDays} days`);
      } else if (!predictionResponse.success) {
        throw new Error('Prediction failed');
      }
    } catch (error) {
      this.handlePredictionError(error);
      if (showPredictionsBtn) {
        this.resetButton(showPredictionsBtn, 'Predict');
      }
    } finally {
      if (showPredictionsBtn) {
        this.resetButton(showPredictionsBtn, 'Predict');
      }
    }
  }

  /**
   * Prepares the UI to show predictions
   * @param {string} ticker - Stock symbol
   * @param {number} forecastDays - Forecast days
   */
  preparePredictionUI(ticker, forecastDays) {
    const { predictionCard, predictionFigure } = this.elements;
    
    // Ensure prediction card is visible
    if (predictionCard) {
      if (predictionCard.classList.contains('d-none')) {
        predictionCard.classList.remove('d-none');
      }
      
      // Update prediction card title
      this.updateCardTitle(predictionCard, `${ticker} Price Predictions (${forecastDays} days)`);
    }
    
    // Show loading indicator in prediction figure
    if (predictionFigure) {
      predictionFigure.classList.remove('d-none');
      this.showLoadingInElement(predictionFigure, 'Generating predictions...');
    }
  }

  /**
   * Handles errors during prediction
   * @param {Error} error - Error occurred
   */
  handlePredictionError(error) {
    const { predictionFigure } = this.elements;
    
    // Hide prediction figure if there is an error
    if (predictionFigure) {
      predictionFigure.classList.add('d-none');
    }
    
    this.showError(`Error getting predictions: ${error.message}`);
  }

  /**
   * Renders predictions in the active view (table or chart)
   */
  renderPredictions() {
    const { predictionFigure, tableDisplayRadio } = this.elements;
    
    if (!this.currentPredictions || !this.currentTicker || !predictionFigure) return;
    
    // Clear previous content
    predictionFigure.innerHTML = '';
    
    if (tableDisplayRadio && tableDisplayRadio.checked) {
      // Render table view
      const tableElement = predictionTable.renderTable(this.currentPredictions);
      predictionFigure.appendChild(tableElement);
    } else {
      // Render chart view
      this.renderChartView();
    }
  }

  /**
   * Renders the chart view for predictions
   */
  renderChartView() {
    const { predictionFigure } = this.elements;
    
    const chartContainer = document.createElement('div');
    chartContainer.id = 'predictionChart';
    chartContainer.style.width = '100%';
    chartContainer.style.height = '100%';
    predictionFigure.appendChild(chartContainer);
    
    // Small delay to ensure container has correct size
    setTimeout(() => {
      predictionChart.renderChart(this.currentPredictions);
    }, 50);
  }

  /**
   * Toggles between table and chart view
   * @param {Event} e - Change event
   */
  togglePredictionDisplay(e) {
    const { predictionFigure } = this.elements;
    
    if (!this.currentPredictions || !this.currentTicker || !predictionFigure) return;
    
    const displayType = e.target.value;
    
    // Show loading indicator
    this.showLoadingInElement(predictionFigure, 'Switching view...');
    
    // Small delay to allow loading spinner to be visible
    setTimeout(() => {
      this.renderPredictions();
    }, 200);
  }

  /**
   * Shows a loading indicator in the specified element
   * @param {HTMLElement} element - Element to show loader in
   * @param {string} message - Loading message
   */
  showLoadingInElement(element, message) {
    element.innerHTML = `
      <div class="loading-container">
        <div class="spinner-border spinner-border-xl" role="status">
        </div>
        <div class="loading-text text-light">${message}</div>
      </div>
    `;
  }

  /**
   * Updates the title of a card
   * @param {HTMLElement} cardElement - Card element
   * @param {string} title - New title
   */
  updateCardTitle(cardElement, title) {
    const cardTitle = cardElement.querySelector('.card-header h5');
    if (cardTitle) {
      cardTitle.textContent = title;
    }
  }

  /**
   * Sets a button to loading state
   * @param {HTMLElement} button - Button element
   * @param {string} loadingText - Text to show while loading
   */
  setButtonLoading(button, loadingText) {
    button.disabled = true;
    button.dataset.originalText = button.innerHTML;
    button.innerHTML = `
      <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
      <span>${loadingText}</span>
    `;
  }

  /**
   * Resets a button to its normal state
   * @param {HTMLElement} button - Button element
   * @param {string} text - Optional text for the button (uses original text by default)
   */
  resetButton(button, text) {
    button.disabled = false;
    button.innerHTML = text || button.dataset.originalText || button.innerHTML;
  }

  /**
   * Shows an error message
   * @param {string} message - Error message to display
   */
  showError(message) {
    const { errorAlert } = this.elements;
    if (errorAlert) {
      errorAlert.textContent = message;
      errorAlert.classList.remove('d-none');
    }
  }

  /**
   * Hides the error message
   */
  hideError() {
    const { errorAlert } = this.elements;
    if (errorAlert && !errorAlert.classList.contains('d-none')) {
      errorAlert.classList.add('d-none');
    }
  }
}

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new StockPredictorApp();
  app.init();
});

export { StockPredictorApp };