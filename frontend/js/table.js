export class predictionTable {
    /**
     * Render a table with prediction data
     * @param {Array} predictions - Array of prediction objects
     * @returns {HTMLElement} - The rendered table element
     */
    static renderTable(predictions) {
        // Create table container
        const tableContainer = document.createElement('div');
        tableContainer.className = 'table-responsive h-100 w-100';
        
        // Create table
        const table = document.createElement('table');
        table.className = 'table table-striped table-hover';
        
        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>Date</th>
            <th>Predicted Price</th>
        `;
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        // Add rows for each prediction
        predictions.forEach(prediction => {
            const row = document.createElement('tr');
            
            // Format date - parse as local date to avoid timezone offset issues
            const dateParts = prediction.predicted_date.split('-');
            const date = new Date(parseInt(dateParts[0]), parseInt(dateParts[1]) - 1, parseInt(dateParts[2]));
            const formattedDate = date.toLocaleDateString('en-US', {
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
            
            // Format price
            const formattedPrice = '$' + parseFloat(prediction.predicted_price).toFixed(2);
            
            row.innerHTML = `
                <td>${formattedDate}</td>
                <td>${formattedPrice}</td>
            `;
            
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);
        
        return tableContainer;
    }
}
