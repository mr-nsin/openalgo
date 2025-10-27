document.addEventListener('DOMContentLoaded', function() {
    // Initialize DOM elements
    const symbolSelect = document.getElementById('symbolSelect');
    const expirySelect = document.getElementById('expirySelect');
    const refreshSymbolsBtn = document.getElementById('refreshSymbolsBtn');
    const showOptionChainBtn = document.getElementById('showOptionChainBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const optionChainTableBody = document.getElementById('optionChainTableBody');
    const selectedSymbolEl = document.getElementById('selectedSymbol');
    const selectedExpiryEl = document.getElementById('selectedExpiry');
    const totalStrikesEl = document.getElementById('totalStrikes');
    const lastUpdatedEl = document.getElementById('lastUpdated');
    const wsStatusDot = document.getElementById('wsStatusDot');
    const wsStatusText = document.getElementById('wsStatusText');

    // State variables
    let currentSymbol = '';
    let currentExpiry = '';
    let optionChainData = {};
    let websocket = null;
    let isSubscribed = false;
    let symbolTokenMap = {};

    // CSRF token
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

    // Initialize
    loadSymbols();
    initializeWebSocket();

    // Event listeners
    symbolSelect.addEventListener('change', handleSymbolChange);
    expirySelect.addEventListener('change', handleExpiryChange);
    refreshSymbolsBtn.addEventListener('click', loadSymbols);
    showOptionChainBtn.addEventListener('click', loadOptionChain);

    // Socket.IO event listeners for backend communication
    if (typeof socket !== 'undefined') {
        socket.on('option_chain_subscription', handleSubscriptionEvent);
        socket.on('option_chain_update', handleMarketDataUpdate);
    }

    async function loadSymbols() {
        try {
            showLoading(refreshSymbolsBtn, 'Loading...');
            
            const response = await fetch('/optionchain/api/symbols');
            const data = await response.json();
            
            if (data.status === 'success') {
                populateSymbolDropdown(data.data);
                showToast('Symbols loaded successfully', 'success');
            } else {
                showToast(data.message || 'Failed to load symbols', 'error');
            }
        } catch (error) {
            console.error('Error loading symbols:', error);
            showToast('Failed to load symbols', 'error');
        } finally {
            hideLoading(refreshSymbolsBtn, 'Refresh Symbols');
        }
    }

    function populateSymbolDropdown(symbols) {
        symbolSelect.innerHTML = '<option value="">Select Symbol</option>';
        symbols.forEach(symbol => {
            const option = document.createElement('option');
            option.value = symbol.symbol;
            option.textContent = symbol.name;
            symbolSelect.appendChild(option);
        });
    }

    async function handleSymbolChange() {
        const selectedSymbol = symbolSelect.value;
        
        if (!selectedSymbol) {
            resetExpiryDropdown();
            resetUI();
            return;
        }

        currentSymbol = selectedSymbol;
        selectedSymbolEl.textContent = selectedSymbol;
        
        try {
            const response = await fetch(`/optionchain/api/expiry?symbol=${encodeURIComponent(selectedSymbol)}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                populateExpiryDropdown(data.data);
            } else {
                showToast(data.message || 'Failed to load expiry dates', 'error');
                resetExpiryDropdown();
            }
        } catch (error) {
            console.error('Error loading expiry dates:', error);
            showToast('Failed to load expiry dates', 'error');
            resetExpiryDropdown();
        }
    }

    function populateExpiryDropdown(expiries) {
        expirySelect.innerHTML = '<option value="">Select Expiry</option>';
        expiries.forEach(expiry => {
            const option = document.createElement('option');
            option.value = expiry.expiry;
            option.textContent = expiry.display;
            expirySelect.appendChild(option);
        });
        expirySelect.disabled = false;
    }

    function handleExpiryChange() {
        const selectedExpiry = expirySelect.value;
        
        if (!selectedExpiry) {
            showOptionChainBtn.disabled = true;
            selectedExpiryEl.textContent = 'No expiry selected';
            return;
        }

        currentExpiry = selectedExpiry;
        selectedExpiryEl.textContent = selectedExpiry;
        showOptionChainBtn.disabled = false;
    }

    async function loadOptionChain() {
        if (!currentSymbol || !currentExpiry) {
            showToast('Please select both symbol and expiry', 'warning');
            return;
        }

        try {
            showLoadingOverlay();
            
            // First, get the option chain data
            const response = await fetch('/optionchain/api/data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    symbol: currentSymbol,
                    expiry: currentExpiry
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                optionChainData = data.data;
                renderOptionChain(optionChainData);
                
                // Subscribe to real-time updates
                await subscribeToUpdates();
                
                showToast('Option chain loaded successfully', 'success');
            } else {
                showToast(data.message || 'Failed to load option chain', 'error');
            }
        } catch (error) {
            console.error('Error loading option chain:', error);
            showToast('Failed to load option chain', 'error');
        } finally {
            hideLoadingOverlay();
        }
    }

    async function subscribeToUpdates() {
        try {
            const response = await fetch('/optionchain/api/subscribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    symbol: currentSymbol,
                    expiry: currentExpiry
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('Subscription initiated:', data.data);
            } else {
                console.error('Subscription failed:', data.message);
            }
        } catch (error) {
            console.error('Error subscribing to updates:', error);
        }
    }

    function renderOptionChain(data) {
        if (!data.options || data.options.length === 0) {
            optionChainTableBody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center py-8">
                        <p class="text-lg opacity-70">No option data available</p>
                    </td>
                </tr>
            `;
            return;
        }

        let html = '';
        data.options.forEach(option => {
            const call = option.call || {};
            const put = option.put || {};
            
            html += `
                <tr data-strike="${option.strike}">
                    <!-- Call Options -->
                    <td class="call-section text-center oi-cell" data-token="${call.token}" data-field="oi">
                        ${call.oi || '-'}
                    </td>
                    <td class="call-section text-center" data-token="${call.token}" data-field="change">
                        <span class="change-cell">${formatChange(call.change)}</span>
                    </td>
                    <td class="call-section text-center ltp-cell" data-token="${call.token}" data-field="ltp">
                        ${call.ltp || '-'}
                    </td>
                    <td class="call-section text-center text-xs">
                        ${call.symbol || '-'}
                    </td>
                    
                    <!-- Strike -->
                    <td class="strike-column">${option.strike}</td>
                    
                    <!-- Put Options -->
                    <td class="put-section text-center text-xs">
                        ${put.symbol || '-'}
                    </td>
                    <td class="put-section text-center ltp-cell" data-token="${put.token}" data-field="ltp">
                        ${put.ltp || '-'}
                    </td>
                    <td class="put-section text-center" data-token="${put.token}" data-field="change">
                        <span class="change-cell">${formatChange(put.change)}</span>
                    </td>
                    <td class="put-section text-center oi-cell" data-token="${put.token}" data-field="oi">
                        ${put.oi || '-'}
                    </td>
                </tr>
            `;
        });

        optionChainTableBody.innerHTML = html;
        totalStrikesEl.textContent = data.options.length;
        updateLastUpdated();
    }

    function formatChange(change) {
        if (!change || change === 0) return '-';
        
        const changeValue = parseFloat(change);
        const className = changeValue >= 0 ? 'change-positive' : 'change-negative';
        const sign = changeValue >= 0 ? '+' : '';
        
        return `<span class="${className}">${sign}${changeValue.toFixed(2)}</span>`;
    }

    function handleSubscriptionEvent(data) {
        console.log('Subscription event received:', data);
        
        if (data.action === 'subscribe') {
            symbolTokenMap = data.symbol_map || {};
            
            // Connect to WebSocket for market data
            if (data.tokens && data.tokens.length > 0) {
                connectToWebSocket(data.tokens);
            }
        } else if (data.action === 'unsubscribe') {
            disconnectWebSocket();
        }
    }

    function handleMarketDataUpdate(data) {
        console.log('Market data update received:', data);
        updateOptionChainCell(data);
    }

    function initializeWebSocket() {
        // This will be used for direct WebSocket connection to the proxy
        updateWebSocketStatus(false);
    }

    function connectToWebSocket(tokens) {
        try {
            // Connect to WebSocket proxy (adjust URL as needed)
            const wsUrl = `ws://${window.location.host}/ws`;
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function() {
                console.log('WebSocket connected');
                updateWebSocketStatus(true);
                
                // Authenticate and subscribe
                websocket.send(JSON.stringify({
                    type: 'auth',
                    token: 'your-auth-token' // This should come from the backend
                }));
                
                // Subscribe to tokens
                websocket.send(JSON.stringify({
                    type: 'subscribe',
                    tokens: tokens
                }));
                
                isSubscribed = true;
            };
            
            websocket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'market_data') {
                        updateOptionChainCell(data);
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            websocket.onclose = function() {
                console.log('WebSocket disconnected');
                updateWebSocketStatus(false);
                isSubscribed = false;
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    if (currentSymbol && currentExpiry) {
                        connectToWebSocket(tokens);
                    }
                }, 5000);
            };
            
            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateWebSocketStatus(false);
            };
            
        } catch (error) {
            console.error('Error connecting to WebSocket:', error);
            updateWebSocketStatus(false);
        }
    }

    function disconnectWebSocket() {
        if (websocket) {
            websocket.close();
            websocket = null;
        }
        isSubscribed = false;
        updateWebSocketStatus(false);
    }

    function updateWebSocketStatus(connected) {
        if (connected) {
            wsStatusDot.classList.add('connected');
            wsStatusText.textContent = 'Connected';
        } else {
            wsStatusDot.classList.remove('connected');
            wsStatusText.textContent = 'Disconnected';
        }
    }

    function updateOptionChainCell(data) {
        const token = data.token;
        if (!token) return;
        
        // Find cells with this token
        const cells = document.querySelectorAll(`[data-token="${token}"]`);
        
        cells.forEach(cell => {
            const field = cell.getAttribute('data-field');
            let value = data[field];
            
            if (field === 'ltp' && value !== undefined) {
                cell.textContent = parseFloat(value).toFixed(2);
            } else if (field === 'oi' && value !== undefined) {
                cell.textContent = parseInt(value).toLocaleString();
            } else if (field === 'change' && value !== undefined) {
                const changeSpan = cell.querySelector('.change-cell');
                if (changeSpan) {
                    changeSpan.innerHTML = formatChange(value);
                }
            }
        });
        
        updateLastUpdated();
    }

    function updateLastUpdated() {
        const now = new Date();
        lastUpdatedEl.textContent = now.toLocaleTimeString();
    }

    function resetExpiryDropdown() {
        expirySelect.innerHTML = '<option value="">Select Expiry</option>';
        expirySelect.disabled = true;
        showOptionChainBtn.disabled = true;
    }

    function resetUI() {
        selectedSymbolEl.textContent = '-';
        selectedExpiryEl.textContent = 'No expiry selected';
        totalStrikesEl.textContent = '0';
        lastUpdatedEl.textContent = '-';
        
        optionChainTableBody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center py-8">
                    <div class="flex flex-col items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        <p class="text-lg opacity-70">Select a symbol and expiry to view option chain</p>
                    </div>
                </td>
            </tr>
        `;
        
        // Disconnect WebSocket
        disconnectWebSocket();
    }

    function showLoadingOverlay() {
        if (loadingOverlay) {
            loadingOverlay.classList.remove('hidden');
        }
    }

    function hideLoadingOverlay() {
        if (loadingOverlay) {
            loadingOverlay.classList.add('hidden');
        }
    }

    function showLoading(button, text) {
        if (button) {
            button.disabled = true;
            button.innerHTML = `<span class="loading loading-spinner loading-sm"></span> ${text}`;
        }
    }

    function hideLoading(button, originalText) {
        if (button) {
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }

    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        disconnectWebSocket();
    });
});
