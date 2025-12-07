"""
Kafka Producer para enviar imágenes al sistema de clasificación de frutas
"""

import json
import base64
import time
import os
from datetime import datetime
from kafka import KafkaProducer
from PIL import Image
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FruitImageProducer:
    def __init__(self, kafka_servers="localhost:9092", topic="fruit-images"):
        """
        Inicializa el producer de Kafka
        
        Args:
            kafka_servers (str): Servidores Kafka
            topic (str): Tópico donde enviar las imágenes
        """
        self.kafka_servers = kafka_servers
        self.topic = topic
        
        # Configurar producer
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_servers],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None
        )
        
        logger.info(f"Producer creado para topic: {topic}")
    
    def send_image(self, image_path, user_id="default_user"):
        """
        Envía una imagen individual al tópico de Kafka
        
        Args:
            image_path (str): Ruta a la imagen
            user_id (str): ID del usuario
        """
        try:
            # Leer y codificar imagen
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Verificar que es una imagen válida
            image = Image.open(BytesIO(image_bytes))
            
            # Redimensionar si es muy grande (optimización)
            if image.size[0] > 1024 or image.size[1] > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
            
            # Codificar en base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Crear mensaje
            message = {
                'image_name': os.path.basename(image_path),
                'image_base64': image_base64,
                'user_id': user_id,
                'upload_time': datetime.now().isoformat()
            }
            
            # Enviar a Kafka
            key = f"{user_id}_{int(time.time())}"
            future = self.producer.send(self.topic, key=key, value=message)
            
            # Confirmar envío
            record_metadata = future.get(timeout=10)
            
            logger.info(f"Imagen enviada: {os.path.basename(image_path)} "
                       f"(Topic: {record_metadata.topic}, "
                       f"Partition: {record_metadata.partition}, "
                       f"Offset: {record_metadata.offset})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error enviando imagen {image_path}: {str(e)}")
            return False
    
    def send_batch_images(self, directory_path, user_id="batch_user", delay=1.0):
        """
        Envía todas las imágenes de un directorio
        
        Args:
            directory_path (str): Directorio con imágenes
            user_id (str): ID del usuario
            delay (float): Segundos de delay entre envíos
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        sent_count = 0
        error_count = 0
        
        logger.info(f"Iniciando envío en lote desde: {directory_path}")
        
        for filename in os.listdir(directory_path):
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in image_extensions:
                image_path = os.path.join(directory_path, filename)
                
                if self.send_image(image_path, user_id):
                    sent_count += 1
                else:
                    error_count += 1
                
                # Delay entre envíos
                if delay > 0:
                    time.sleep(delay)
        
        logger.info(f"Envío completado - Exitosos: {sent_count}, Errores: {error_count}")
        return sent_count, error_count
    
    def send_continuous_stream(self, directory_path, user_id="stream_user", 
                             interval=5.0, loop=True):
        """
        Envía imágenes continuamente para simular un stream en tiempo real
        
        Args:
            directory_path (str): Directorio con imágenes
            user_id (str): ID del usuario
            interval (float): Segundos entre envíos
            loop (bool): Si debe repetir las imágenes cuando termine
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in os.listdir(directory_path) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not image_files:
            logger.error(f"No se encontraron imágenes en {directory_path}")
            return
        
        logger.info(f"Iniciando stream continuo con {len(image_files)} imágenes")
        
        try:
            while True:
                for filename in image_files:
                    image_path = os.path.join(directory_path, filename)
                    
                    # Agregar timestamp al user_id para simular usuarios diferentes
                    stream_user_id = f"{user_id}_{int(time.time())}"
                    
                    self.send_image(image_path, stream_user_id)
                    
                    logger.info(f"Próxima imagen en {interval} segundos...")
                    time.sleep(interval)
                
                if not loop:
                    break
                    
                logger.info("Reiniciando ciclo de imágenes...")
                
        except KeyboardInterrupt:
            logger.info("Stream interrumpido por usuario")
    
    def close(self):
        """Cierra el producer"""
        if self.producer:
            self.producer.close()
            logger.info("Producer cerrado")


def main():
    """Función principal del producer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Producer para Imágenes de Frutas')
    parser.add_argument('--mode', choices=['single', 'batch', 'stream'], required=True,
                       help='Modo de envío')
    parser.add_argument('--image', help='Ruta a imagen individual (modo single)')
    parser.add_argument('--directory', help='Directorio con imágenes (modo batch/stream)')
    parser.add_argument('--user-id', default='test_user', help='ID del usuario')
    parser.add_argument('--kafka-servers', default='localhost:9092', help='Servidores Kafka')
    parser.add_argument('--topic', default='fruit-images', help='Tópico de Kafka')
    parser.add_argument('--delay', type=float, default=1.0, 
                       help='Delay entre envíos (segundos)')
    parser.add_argument('--interval', type=float, default=5.0,
                       help='Intervalo para modo stream (segundos)')
    parser.add_argument('--no-loop', action='store_true',
                       help='No repetir en modo stream')
    
    args = parser.parse_args()
    
    # Crear producer
    producer = FruitImageProducer(
        kafka_servers=args.kafka_servers,
        topic=args.topic
    )
    
    try:
        if args.mode == 'single':
            if not args.image:
                print("Error: Para modo single necesitas --image")
                return
            
            success = producer.send_image(args.image, args.user_id)
            print(f"Envío {'exitoso' if success else 'fallido'}")
            
        elif args.mode == 'batch':
            if not args.directory:
                print("Error: Para modo batch necesitas --directory")
                return
            
            sent, errors = producer.send_batch_images(
                args.directory, args.user_id, args.delay
            )
            print(f"Enviadas: {sent}, Errores: {errors}")
            
        elif args.mode == 'stream':
            if not args.directory:
                print("Error: Para modo stream necesitas --directory")
                return
            
            producer.send_continuous_stream(
                args.directory, args.user_id, args.interval, not args.no_loop
            )
            
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
    finally:
        producer.close()


if __name__ == "__main__":
    main()