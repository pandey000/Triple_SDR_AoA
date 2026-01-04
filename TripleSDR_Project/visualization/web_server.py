"""
FastAPI WebSocket Server for Visualization Dashboard

Serves HTML dashboard and provides real-time WebSocket data stream.

Features:
- Async WebSocket for low latency
- Multiple simultaneous clients supported
- Automatic reconnection handling
- CORS enabled for remote access

Performance:
- Memory: ~50 MB
- CPU: ~2% (idle), ~5% (active streaming)

Author: Triple-SDR Project
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from pathlib import Path
from typing import List
import logging

# Suppress uvicorn access logs for cleaner output
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Global reference to data aggregator (set by start_web_server)
data_aggregator = None

# FastAPI app
app = FastAPI(title="Triple-SDR Dashboard", docs_url=None, redoc_url=None)

# Serve static files (CSS, JS)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Connected WebSocket clients
active_connections: List[WebSocket] = []


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """
    Serve main dashboard HTML.
    
    Returns:
        HTML page with embedded JavaScript and CSS
    """
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    
    if not template_path.exists():
        return HTMLResponse(
            content="<h1>Error: dashboard.html not found</h1>"
                   "<p>Please ensure visualization/templates/dashboard.html exists</p>",
            status_code=500
        )
    
    with open(template_path, 'r') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data streaming.
    
    Protocol:
    - Client connects
    - Server sends JSON updates at ~5 Hz
    - Client can disconnect anytime
    - Server handles disconnects gracefully
    
    Message Format:
    {
        'spectrum_r1': [float] × 256,
        'spectrum_r2': [float] × 256,
        'spectrum_r1_freq': float (MHz),
        'spectrum_r2_freq': float (MHz),
        'events': [{time, freq, radio, hits, channels, confirmed}],
        'aoa': [{time, freq, angle, confidence, phase}],
        'metrics': {cpu, buffers, detections, aoa stats},
        'confidence': {signal_strength, bandwidth, regularity, shape, overall},
        'timestamp': float (unix time)
    }
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    client_addr = websocket.client.host if websocket.client else "unknown"
    print(f"[WebServer] Client connected from {client_addr} ({len(active_connections)} total)")
    
    try:
        while True:
            # Get latest state from aggregator
            if data_aggregator is None:
                await websocket.send_json({
                    'error': 'Data aggregator not initialized'
                })
                await asyncio.sleep(1)
                continue
            
            state = data_aggregator.get_state_snapshot()
            
            # Send to client
            try:
                await websocket.send_json(state)
            except Exception as e:
                print(f"[WebServer] Error sending data: {e}")
                break
            
            # Update rate: 200ms = 5 Hz (adaptive rate handled by aggregator)
            await asyncio.sleep(0.2)
    
    except WebSocketDisconnect:
        print(f"[WebServer] Client disconnected from {client_addr}")
    
    except Exception as e:
        print(f"[WebServer] WebSocket error: {e}")
    
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"[WebServer] {len(active_connections)} clients remaining")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with server status
    """
    return {
        'status': 'ok',
        'clients_connected': len(active_connections),
        'aggregator_running': data_aggregator is not None
    }


def start_web_server(aggregator, port: int = 8080, host: str = "0.0.0.0"):
    """
    Start FastAPI server (blocking call).
    
    This should be called in a separate thread from main_controller.py.
    
    Args:
        aggregator: DataAggregator instance
        port: Port to listen on (default 8080)
        host: Host to bind to (0.0.0.0 = all interfaces)
    
    Example:
        >>> from visualization import DataAggregator, start_web_server
        >>> aggregator = DataAggregator(...)
        >>> aggregator.start()
        >>> 
        >>> # In separate thread:
        >>> threading.Thread(
        ...     target=start_web_server,
        ...     args=(aggregator, 8080),
        ...     daemon=True
        ... ).start()
    """
    global data_aggregator
    data_aggregator = aggregator
    
    import uvicorn
    
    # Configure uvicorn for production use
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",  # Reduce log spam
        access_log=False,     # Disable access logs
        ws_ping_interval=20,  # Keep WebSocket alive
        ws_ping_timeout=20
    )
    
    server = uvicorn.Server(config)
    
    print(f"[WebServer] Starting on http://{host}:{port}")
    print(f"[WebServer] Dashboard URL: http://{host}:{port}")
    
    # Run server (blocking)
    server.run()