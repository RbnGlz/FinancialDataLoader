# FinancialDataLoader Documentación

## Descripción General

FinancialDataLoader es una clase Python diseñada para la descarga, almacenamiento y visualización eficiente de datos financieros. Está optimizada para trabajar con múltiples fuentes de datos y proporciona herramientas para el procesamiento en paralelo y la visualización de series temporales financieras.

### Características Principales
- **Descarga de datos financieros desde múltiples fuentes:** Yahoo Finance, Stooq.
- **Sistema de caché avanzado:** Evita descargas repetidas mediante la gestión de archivos y memoria caché.
- **Procesamiento en paralelo:** Mejora la eficiencia utilizando ThreadPoolExecutor o ProcessPoolExecutor.
- **Visualización personalizable:** Permite graficar precios y otros indicadores con opciones avanzadas.
- **Manejo robusto de errores y validación de parámetros:** Asegura la estabilidad y confiabilidad del código.

---

## Código

El siguiente es el código completo de la clase `DataLoader` junto con funciones auxiliares:

```python
"""
Contiene la clase DataLoader que implementa:
    Descarga de datos financieros con caché.
    Procesamiento en paralelo.
    Funciones de visualización de precios.
"""   
# Importar librerías
import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from typing import List, Optional, Dict, Tuple, Union
import yfinance as yf
import os
import pickle
import multiprocessing
from pathlib import Path
import functools
import time
import hashlib
from tqdm.auto import tqdm

# Configuración de logging para ver mensajes informativos en consola
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Decorador para medir el tiempo de ejecución de funciones
def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Función {func.__name__} ejecutada en {end_time - start_time:.4f} segundos")
        return result
    return wrapper

# Clase DataLoader
class DataLoader:
    """
    Clase para cargar y procesar datos financieros de múltiples fuentes.
    Características:
    - Caché de datos para evitar descargas repetidas
    - Procesamiento en paralelo para mejorar la eficiencia
    - Visualización de datos financieros
    """
    # Fuentes de datos soportadas
    SUPPORTED_SOURCES = {"yahoo", "stooq"}
    
    def __init__(self, fecha_inicio: str = "2020-01-01", 
                 fecha_final: str = None,
                 cache_dir: str = "data_cache",
                 use_cache: bool = True,
                 cache_expiry_days: int = 1):
        """
        Inicializa el cargador de datos financieros.
        Args:
            fecha_inicio: Fecha de inicio para los datos (formato "YYYY-MM-DD").
            fecha_final: Fecha final para los datos (formato "YYYY-MM-DD"). 
                         Si es None, se usa la fecha actual.
            cache_dir: Directorio para almacenar la caché de datos.
            use_cache: Si es True, se utilizará la caché para evitar descargas repetidas.
            cache_expiry_days: Número de días tras los cuales la caché se considera expirada.
        """
        # Configurar fecha final como la fecha actual si no se proporciona
        if fecha_final is None:
            fecha_final = datetime.now().strftime("%Y-%m-%d")
        # Guardamos ambas representaciones: string y datetime
        self.fecha_inicio_str = fecha_inicio
        self.fecha_final_str = fecha_final
        self.fecha_inicio = pd.to_datetime(fecha_inicio)
        self.fecha_final = pd.to_datetime(fecha_final)
        # Validar fechas
        if self.fecha_inicio > self.fecha_final:
            raise ValueError("La fecha de inicio debe ser anterior a la fecha final")
        # Configuración de caché
        self.use_cache = use_cache
        self.cache_expiry_days = cache_expiry_days
        # Convertimos el directorio de caché a objeto Path y lo creamos
        self.cache_dir = Path(cache_dir)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Diccionario para almacenar datos en memoria
        self._memory_cache = {}
    
    def _get_cache_filename(self, ticker: str, source: str) -> str:
        """
        Genera un nombre de archivo único para la caché basado en los parámetros.
        Usa un hash para manejar caracteres especiales en los tickers.
        """
        # Crear un hash para evitar problemas con caracteres especiales en nombres de archivo
        params_str = f"{ticker}_{source}_{self.fecha_inicio_str}_{self.fecha_final_str}"
        filename_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"{ticker}_{source}_{filename_hash}.pkl"
    
    def _get_cache_path(self, ticker: str, source: str) -> Path:
        """
        Genera la ruta del archivo de caché para un ticker y fuente específicos.
        """
        filename = self._get_cache_filename(ticker, source)
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Verifica si un archivo de caché es válido basado en su fecha de modificación.
        Returns:
            True si la caché es válida, False si ha expirado o no existe.
        """
        if not cache_path.exists():
            return False
        # Verificar si la caché ha expirado
        if self.cache_expiry_days > 0:
            mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            expiry_time = datetime.now() - timedelta(days=self.cache_expiry_days)
            if mod_time < expiry_time:
                logger.debug(f"Caché expirada para {cache_path.name}")
                return False
        return True
    
    def _load_from_cache(self, ticker: str, source: str) -> Optional[pd.DataFrame]:
        """
        Intenta cargar datos desde la caché.
        Primero verifica la caché en memoria, luego la caché en disco.
        Returns:
            DataFrame con los datos si la caché es válida, None en caso contrario.
        """
        if not self.use_cache:
            return None
        # Verificar caché en memoria
        memory_key = f"{ticker}_{source}"
        if memory_key in self._memory_cache:
            logger.debug(f"Datos cargados desde caché en memoria para {ticker} de {source}")
            return self._memory_cache[memory_key]
        # Verificar caché en disco
        cache_path = self._get_cache_path(ticker, source)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                # Almacenar en caché de memoria para acceso más rápido
                self._memory_cache[memory_key] = data
                logger.info(f"Datos cargados desde caché en disco para {ticker} de {source}")
                return data
            except (pickle.PickleError, EOFError, AttributeError) as e:
                logger.warning(f"Error al cargar caché para {ticker}: {e}")
                # Eliminar archivo de caché corrupto
                cache_path.unlink(missing_ok=True)
        return None
    
    def _save_to_cache(self, ticker: str, source: str, data: pd.DataFrame) -> None:
        """
        Guarda los datos en caché (memoria y disco).
        """
        if not self.use_cache or data is None or data.empty:
            return
        # Guardar en caché de memoria
        memory_key = f"{ticker}_{source}"
        self._memory_cache[memory_key] = data
        # Guardar en caché de disco
        cache_path = self._get_cache_path(ticker, source)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Datos guardados en caché para {ticker} de {source}")
        except Exception as e:
            logger.warning(f"Error al guardar caché para {ticker}: {e}")
    
    @timer_decorator
    def _get_data_from_source(self, ticker: str, source: str) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de una fuente específica, utilizando caché si está disponible.
        Args:
            ticker: Símbolo del activo financiero.
            source: Fuente de datos ("yahoo" o "stooq").
        Returns:
            DataFrame con los datos financieros o None si hay error.
        """
        # Validar fuente
        if source not in self.SUPPORTED_SOURCES:
            logger.error(f"Fuente {source} no soportada. Fuentes disponibles: {self.SUPPORTED_SOURCES}")
            return None
        # Intentar cargar datos desde la caché
        cached_data = self._load_from_cache(ticker, source)
        if cached_data is not None:
            return cached_data
        try:
            if source == "stooq":
                df = pdr.get_data_stooq(symbols=ticker, 
                                        start=self.fecha_inicio, 
                                        end=self.fecha_final)
            elif source == "yahoo":
                df = yf.download(ticker, 
                                 start=self.fecha_inicio, 
                                 end=self.fecha_final,
                                 progress=False)
            else:
                return None  # Ya validamos antes, pero por si acaso
            # Verificar si se obtuvieron datos
            if df is None or df.empty:
                logger.warning(f"No se encontraron datos para {ticker} en {source}")
                return None
            # Ordenar el DataFrame cronológicamente usando el índice (fechas)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            # Guardar en caché para evitar descargas futuras innecesarias
            self._save_to_cache(ticker, source, df)
            return df
        except ConnectionError as ce:
            logger.error(f"Error de conexión al descargar datos para {ticker}: {ce}")
            raise ConnectionError(f"No se pudo conectar al servicio de datos: {ce}")
        except ValueError as ve:
            logger.error(f"Error de valor al descargar datos para {ticker}: {ve}")
            raise
        except Exception as error:
            logger.error(f"Error inesperado al obtener datos de {source} para {ticker}: {error}")
            return None
    
    def _process_ticker(self, args: Tuple[str, str, bool]) -> Optional[Tuple[str, pd.DataFrame]]:
        """
        Función auxiliar para el procesamiento en paralelo.
        Args:
            args: Una tupla que contiene (ticker, fuente, show_progress).
        Returns:
            Una tupla (clave, DataFrame) si la descarga es exitosa, o None en caso de error.
        """
        ticker, source, _ = args
        df = self._get_data_from_source(ticker, source)
        if df is not None:
            return (f"{ticker}_{source}", df)
        return None
    
    @timer_decorator
    def load_parallel(self, tickers: List[str], 
                      sources: List[str] = None,
                      executor_type: str = "thread",
                      show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carga datos en paralelo usando un pool de hilos o procesos.
        Args:
            tickers: Lista de símbolos (tickers) a descargar.
            sources: Lista de fuentes desde las cuales se descargan los datos.
                    Si es None, se usan todas las fuentes soportadas.
            executor_type: "thread" para ThreadPoolExecutor o "process" para ProcessPoolExecutor.
            show_progress: Si es True, muestra una barra de progreso.
        Returns:
            Un diccionario donde la clave es el ticker concatenado con la fuente y el valor es el DataFrame.
        """
        # Usar todas las fuentes soportadas si no se especifica
        if sources is None:
            sources = list(self.SUPPORTED_SOURCES)
        else:
            # Filtrar fuentes no soportadas
            sources = [s for s in sources if s in self.SUPPORTED_SOURCES]
            if not sources:
                logger.error(f"Ninguna fuente especificada es soportada. Fuentes disponibles: {self.SUPPORTED_SOURCES}")
                return {}
        results = {}
        # Se genera la lista de tareas para cada combinación de ticker y fuente
        tasks = [(ticker, source, show_progress) for ticker in tickers for source in sources]
        # Determinar el número óptimo de trabajadores
        num_cpus = multiprocessing.cpu_count()
        num_workers = min(num_cpus, len(tasks))
        # Seleccionar el tipo de ejecutor según el argumento
        if executor_type.lower() == "process":
            Executor = ProcessPoolExecutor
            logger.info(f"Usando ProcessPoolExecutor con {num_workers} workers")
        else:
            Executor = ThreadPoolExecutor
            logger.info(f"Usando ThreadPoolExecutor con {num_workers} workers")
        # Ejecutar las tareas en paralelo
        with Executor(max_workers=num_workers) as executor:
            # Usar tqdm para mostrar progreso si se solicita
            if show_progress:
                futures = list(tqdm(executor.map(self._process_ticker, tasks), 
                                   total=len(tasks), 
                                   desc="Descargando datos"))
            else:
                futures = list(executor.map(self._process_ticker, tasks))
            for result in futures:
                if result:
                    key, df = result
                    results[key] = df
        logger.info(f"Datos cargados exitosamente para {len(results)}/{len(tasks)} combinaciones")
        return results
    
    def load_single(self, ticker: str, source: str = "yahoo") -> Optional[pd.DataFrame]:
        """
        Carga datos para un solo ticker y fuente.
        Es un método de conveniencia para cuando solo se necesita un activo.
        Args:
            ticker: Símbolo del activo financiero.
            source: Fuente de datos ("yahoo" o "stooq").
        Returns:
            DataFrame con los datos financieros o None si hay error.
        """
        if source not in self.SUPPORTED_SOURCES:
            logger.error(f"Fuente {source} no soportada. Fuentes disponibles: {self.SUPPORTED_SOURCES}")
            return None
        return self._get_data_from_source(ticker, source)
    
    def plot_prices(self, data: Dict[str, pd.DataFrame], 
                    column: str = "Close",
                    ma_periods: List[int] = None,
                    figsize: Tuple[int, int] = (22, 12),
                    normalize: bool = False) -> None:
        """
        Grafica los precios de múltiples activos con opciones avanzadas.
        Args:
            data: Diccionario de DataFrames con datos financieros.
            column: Columna que se desea graficar (por defecto "Close").
            ma_periods: Lista de períodos para calcular medias móviles.
            figsize: Tamaño de la figura (ancho, alto).
            normalize: Si es True, normaliza los precios para facilitar la comparación.
        """
        if not data:
            logger.warning("No hay datos para graficar")
            return
        # Se establece un estilo moderno para el gráfico
        plt.style.use('seaborn')
        plt.figure(figsize=figsize)
        # Preparar los datos para graficar
        plot_data = {}
        for key, df in data.items():
            if column in df.columns:
                series = df[column].copy()
                # Normalizar los datos si se solicita
                if normalize:
                    series = series / series.iloc[0] * 100
                plot_data[key] = series
            else:
                logger.warning(f"La columna '{column}' no se encontró en los datos de {key}")
        # Graficar los datos
        for key, series in plot_data.items():
            series.plot(label=key, linewidth=2, alpha=0.8)
            # Añadir medias móviles si se solicitan
            if ma_periods:
                for period in ma_periods:
                    ma = series.rolling(window=period).mean()
                    ma.plot(label=f"{key} MA{period}", linestyle='--', alpha=0.6)
        # Configurar el gráfico
        title = f"Precios de {column}"
        if normalize:
            title += " (Normalizados, Base 100)"
            plt.ylabel("Precio Normalizado (Base 100)", size=20)
        else:
            plt.ylabel("Precio", size=20)
        plt.title(title, size=25, pad=20)
        plt.xlabel("Fecha", size=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def clear_cache(self, tickers: List[str] = None, sources: List[str] = None) -> None:
        """
        Limpia la caché para los tickers y fuentes especificados.
        Si no se especifican tickers o fuentes, se limpia toda la caché.
        Args:
            tickers: Lista de tickers para los que se limpiará la caché.
            sources: Lista de fuentes para las que se limpiará la caché.
        """
        # Limpiar caché en memoria
        if tickers is None and sources is None:
            self._memory_cache.clear()
            logger.info("Caché en memoria limpiada completamente")
        else:
            # Limpiar selectivamente
            if tickers is not None and sources is not None:
                keys_to_remove = [f"{ticker}_{source}" for ticker in tickers for source in sources]
            elif tickers is not None:
                keys_to_remove = [k for k in self._memory_cache.keys() if any(k.startswith(f"{ticker}_") for ticker in tickers)]
            else:  # sources is not None
                keys_to_remove = [k for k in self._memory_cache.keys() if any(f"_{source}" in k for source in sources)]
            for key in keys_to_remove:
                if key in self._memory_cache:
                    del self._memory_cache[key]
            logger.info(f"Se eliminaron {len(keys_to_remove)} entradas de la caché en memoria")
        # Limpiar caché en disco
        if not self.use_cache or not self.cache_dir.exists():
            return
        if tickers is None and sources is None:
            # Eliminar todos los archivos de caché
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Caché en disco limpiada completamente")
        else:
            # Eliminar archivos selectivamente
            count = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                filename = cache_file.name
                should_delete = False
                if tickers is not None and sources is not None:
                    should_delete = any(f"{ticker}_{source}_" in filename for ticker in tickers for source in sources)
                elif tickers is not None:
                    should_delete = any(f"{ticker}_" in filename for ticker in tickers)
                else:  # sources is not None
                    should_delete = any(f"_{source}_" in filename for source in sources)
                if should_delete:
                    cache_file.unlink()
                    count += 1
            logger.info(f"Se eliminaron {count} archivos de caché en disco")

def setup_notebook():
    """
    Configura el entorno del notebook para mostrar gráficos inline.
    Esto es útil cuando se ejecuta en Jupyter.
    """
    try:
        import IPython
        IPython.get_ipython().run_line_magic('matplotlib', 'inline')
        plt.style.use('seaborn')
        logger.info("Entorno de notebook configurado correctamente")
    except (ImportError, AttributeError):
        logger.debug("No se está ejecutando en un entorno de notebook")

def main():
    """
    Ejemplo de uso del DataLoader.
    Se configura el entorno (en caso de estar en un notebook), se descargan datos de varios
    tickers en paralelo y se grafican los precios de cierre.
    """
    # Configurar el entorno del notebook
    setup_notebook()
    # Inicializar el cargador de datos con fecha final automática (hoy)
    loader = DataLoader(fecha_inicio="2020-01-01", fecha_final=None, cache_expiry_days=1)
    # Definir una lista de tickers a procesar
    tickers = ["AMZN", "AAPL", "MSFT", "GOOGL", "META"]
    # Cargar datos en paralelo utilizando ThreadPoolExecutor (por defecto para operaciones IO-bound)
    logger.info("Iniciando carga de datos...")
    data = loader.load_parallel(tickers, executor_type="thread", show_progress=True)
    logger.info(f"Datos cargados exitosamente para {len(data)} combinaciones de ticker y fuente")
    # Graficar los precios de cierre
    loader.plot_prices(data, ma_periods=[20, 50])
    # Graficar los precios normalizados para comparación
    loader.plot_prices(data, normalize=True)
    # Ejemplo de limpieza de caché
    # loader.clear_cache(tickers=["AAPL"])

if __name__ == "__main__":
    main()