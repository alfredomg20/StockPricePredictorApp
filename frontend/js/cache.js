/**
 * Cache module for managing localStorage with TTL support
 */
class Cache {
  constructor() {
    this.prefix = 'stockPredictor_';
  }

  /**
   * Sets a value in cache
   * @param {string} key - Cache key
   * @param {*} value - Value to cache
   * @param {number} ttlMs - Time to live in milliseconds (optional)
   */
  set(key, value, ttlMs = null) {
    const cacheKey = this.prefix + key;
    const cacheData = {
      value: value,
      timestamp: Date.now(),
      ttl: ttlMs
    };
    
    try {
      localStorage.setItem(cacheKey, JSON.stringify(cacheData));
    } catch (error) {
      console.warn('Failed to cache data:', error);
    }
  }

  /**
   * Gets a value from cache
   * @param {string} key - Cache key
   * @returns {*} Cached value or null if not found/expired
   */
  get(key) {
    const cacheKey = this.prefix + key;
    
    try {
      const cachedItem = localStorage.getItem(cacheKey);
      if (!cachedItem) return null;
      
      const cacheData = JSON.parse(cachedItem);
      
      // Check if item has expired
      if (cacheData.ttl && (Date.now() - cacheData.timestamp) > cacheData.ttl) {
        this.delete(key);
        return null;
      }
      
      return cacheData.value;
    } catch (error) {
      console.warn('Failed to retrieve cached data:', error);
      return null;
    }
  }

  /**
   * Deletes a value from cache
   * @param {string} key - Cache key
   */
  delete(key) {
    const cacheKey = this.prefix + key;
    try {
      localStorage.removeItem(cacheKey);
    } catch (error) {
      console.warn('Failed to delete cached data:', error);
    }
  }

  /**
   * Clears all cache entries for this app
   */
  clear() {
    try {
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith(this.prefix)) {
          localStorage.removeItem(key);
        }
      });
    } catch (error) {
      console.warn('Failed to clear cache:', error);
    }
  }

  /**
   * Gets cache key for trained model
   * @param {string} ticker - Stock ticker
   * @param {number} forecastDays - Forecast days
   * @returns {string} Cache key
   */
  getTrainedModelKey(ticker, forecastDays) {
    return `trainedModel_${ticker}_${forecastDays}`;
  }

  /**
   * Gets cache key for model metrics
   * @param {string} ticker - Stock ticker
   * @param {number} forecastDays - Forecast days
   * @returns {string} Cache key
   */
  getMetricsKey(ticker, forecastDays) {
    return `metrics_${ticker}_${forecastDays}`;
  }

  /**
   * Gets cache key for predictions
   * @param {string} ticker - Stock ticker
   * @param {number} forecastDays - Forecast days
   * @returns {string} Cache key
   */
  getPredictionsKey(ticker, forecastDays) {
    return `predictions_${ticker}_${forecastDays}`;
  }
}

// Create and export singleton instance
const cache = new Cache();
export { cache };
