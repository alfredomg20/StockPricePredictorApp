import {API_BASE_URL, TICKERS} from './config.js';

// Class to handle price predictor API requests
export class pricePredictorAPI {

    static getAvailableTickers() {
        return TICKERS;
    }

    static async startTraining(ticker, forecastDays) {
        try {
            const response = await fetch(`${API_BASE_URL}/train/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ ticker, forecast_days: forecastDays })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error starting training:', error);
            throw error;
        }
    }

    static async checkTrainingStatus(taskId) {
        try {
            const response = await fetch(`${API_BASE_URL}/train/${taskId}/status`);

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting training status:', error);
            throw error;
        }
    }

    static async getTrainingResult(taskId) {
        try {
            const response = await fetch(`${API_BASE_URL}/train/${taskId}/result`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error getting training result:', error);
            throw error;
        }
    }

    static async makePrediction(ticker, forecastDays) {
        try {
            const response = await fetch(`${API_BASE_URL}/predict/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ ticker, forecast_days: forecastDays })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error making prediction:', error);
            throw error;
        }
    }

    static async getModel(ticker, forecastDays, las_trained_time) {
        try {
            const response = await fetch(`${API_BASE_URL}/model/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ ticker, forecast_days: forecastDays, last_train_date: las_trained_time })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting model:', error);
            throw error;
        }
    }
}
