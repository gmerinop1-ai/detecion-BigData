"""
Script de prueba para el sistema Spark + Kafka de clasificación de frutas
"""

import os
import time
import subprocess
import sys
from pathlib import Path

def check_prerequisites():
    """Verifica que las dependencias estén instaladas"""
    required_packages = ['pyspark', 'kafka-python', 'tf_keras', 'tensorflow', 'pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - FALTANTE")
    
    if missing_packages:
        print(f"\n🔧 Instalar dependencias faltantes:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_directories():
    """Crear directorios necesarios"""
    directories = [
        'test_images',
        'output',
        'output/streaming_results',
        'output/batch_results', 
        'checkpoint',
        'checkpoint/streaming',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directorio creado: {directory}")

def create_sample_images():
    """Crear imágenes de muestra si no existen"""
    from PIL import Image
    import numpy as np
    
    test_dir = Path('test_images')
    
    if not any(test_dir.glob('*.jpg')):
        print("🖼️ Creando imágenes de muestra...")
        
        # Crear imágenes sintéticas de colores que representen frutas
        colors = {
            'apple_red.jpg': (255, 0, 0),      # Rojo para manzana
            'banana_yellow.jpg': (255, 255, 0), # Amarillo para banana  
            'orange_orange.jpg': (255, 165, 0), # Naranja para naranja
            'grape_purple.jpg': (128, 0, 128),  # Púrpura para uvas
            'kiwi_green.jpg': (0, 255, 0)       # Verde para kiwi
        }
        
        for filename, color in colors.items():
            # Crear imagen sintética
            image = Image.new('RGB', (224, 224), color)
            image.save(test_dir / filename)
            print(f"   ✅ Creada: {filename}")
    else:
        print("✅ Imágenes de muestra ya existen")

def test_batch_processing():
    """Probar procesamiento en lote"""
    print("\n🔄 Probando procesamiento en lote...")
    
    cmd = [
        sys.executable, 'spark_consumer.py',
        '--mode', 'batch',
        '--model', 'FV_Fruits_Only.h5',
        '--input', 'test_images',
        '--output', 'output/batch_test'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Procesamiento en lote exitoso")
            print(f"📄 Output: {result.stdout}")
        else:
            print("❌ Error en procesamiento en lote")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout en procesamiento en lote")
    except FileNotFoundError:
        print("❌ No se encontró spark_consumer.py")

def test_kafka_producer():
    """Probar el producer de Kafka"""
    print("\n📤 Probando Kafka Producer...")
    
    cmd = [
        sys.executable, 'kafka_producer.py',
        '--mode', 'single',
        '--image', str(next(Path('test_images').glob('*.jpg'), None)),
        '--user-id', 'test_user'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Producer de Kafka funcionando")
        else:
            print("❌ Error en Producer de Kafka")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout en producer")
    except FileNotFoundError:
        print("❌ No se encontró kafka_producer.py")

def check_model_file():
    """Verificar que el modelo existe"""
    model_path = 'FV_Fruits_Only.h5'
    
    if os.path.exists(model_path):
        print(f"✅ Modelo encontrado: {model_path}")
        return True
    else:
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   💡 Asegúrate de que el modelo esté en el directorio actual")
        return False

def check_kafka_running():
    """Verificar si Kafka está ejecutándose (Windows)"""
    print("🔍 Verificando servicios...")
    
    # En Windows, simplemente informamos cómo verificar
    print("💡 Para verificar Kafka manualmente:")
    print("   - Kafka debe estar ejecutándose en localhost:9092")
    print("   - Crear tópico: kafka-topics.bat --create --topic fruit-images --bootstrap-server localhost:9092")
    
def main():
    """Función principal de prueba"""
    print("🍎 Sistema de Clasificación de Frutas - Spark + Kafka")
    print("=" * 60)
    
    # Verificar prerequisitos
    print("\n1️⃣ Verificando dependencias...")
    if not check_prerequisites():
        print("❌ Dependencias faltantes. Instálalas e intenta de nuevo.")
        return
    
    # Configurar directorios
    print("\n2️⃣ Configurando directorios...")
    setup_directories()
    
    # Verificar modelo
    print("\n3️⃣ Verificando modelo...")
    if not check_model_file():
        print("❌ Modelo requerido no encontrado.")
        return
    
    # Crear imágenes de muestra
    print("\n4️⃣ Preparando imágenes de muestra...")
    create_sample_images()
    
    # Verificar Kafka
    print("\n5️⃣ Información sobre Kafka...")
    check_kafka_running()
    
    # Probar procesamiento en lote
    print("\n6️⃣ Probando sistema...")
    test_batch_processing()
    
    # Probar producer (solo si Kafka está disponible)
    # test_kafka_producer()
    
    print("\n✅ Pruebas completadas!")
    print("\n🚀 Para usar el sistema completo:")
    print("   1. Inicia Kafka y crea el tópico 'fruit-images'")
    print("   2. Ejecuta: python spark_consumer.py --mode streaming --model FV_Fruits_Only.h5")
    print("   3. En otra terminal: python kafka_producer.py --mode batch --directory test_images")

if __name__ == "__main__":
    main()