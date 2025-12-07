"""
Apache Spark Consumer para Clasificación de Frutas con MobileNetV2
Procesa imágenes en tiempo real usando Spark Streaming y Kafka
"""

import os
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import tf_keras as keras
from tf_keras.preprocessing.image import img_to_array
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkFruitClassifier:
    def __init__(self, model_path, kafka_servers="localhost:9092", topic="fruit-images"):
        """
        Inicializa el consumer de Spark para clasificación de frutas
        
        Args:
            model_path (str): Ruta al modelo entrenado (.h5)
            kafka_servers (str): Servidores Kafka
            topic (str): Tópico de Kafka para recibir imágenes
        """
        self.model_path = model_path
        self.kafka_servers = kafka_servers
        self.topic = topic
        
        # Labels de frutas
        self.labels = {
            0: 'apple', 1: 'banana', 2: 'bell pepper', 3: 'chilli pepper', 
            4: 'grapes', 5: 'jalepeno', 6: 'kiwi', 7: 'lemon', 
            8: 'mango', 9: 'orange', 10: 'paprika', 11: 'pear', 
            12: 'pineapple', 13: 'pomegranate', 14: 'watermelon'
        }
        
        # Precios en soles peruanos
        self.precios_soles = {
            'apple': 6.50, 'banana': 2.80, 'bell pepper': 4.50, 'chilli pepper': 8.00,
            'grapes': 9.50, 'jalepeno': 7.50, 'kiwi': 12.00, 'lemon': 3.50,
            'mango': 5.50, 'orange': 4.00, 'paprika': 5.00, 'pear': 7.00,
            'pineapple': 6.00, 'pomegranate': 15.00, 'watermelon': 2.50
        }
        
        # Crear sesión de Spark
        self.spark = self._create_spark_session()
        
        # Broadcast del modelo y configuraciones
        self.broadcast_model_path = self.spark.sparkContext.broadcast(model_path)
        self.broadcast_labels = self.spark.sparkContext.broadcast(self.labels)
        self.broadcast_precios = self.spark.sparkContext.broadcast(self.precios_soles)
        
    def _create_spark_session(self):
        """Crea y configura la sesión de Spark"""
        return SparkSession.builder \
            .appName("FruitClassificationConsumer") \
            .config("spark.sql.streaming.checkpointLocation", "./checkpoint") \
            .config("spark.jars.packages", 
                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
            .getOrCreate()
    
    def process_batch_images(self, input_path, output_path):
        """
        Procesa imágenes en lote usando Spark
        
        Args:
            input_path (str): Directorio con imágenes a procesar
            output_path (str): Directorio para guardar resultados
        """
        logger.info(f"Iniciando procesamiento en lote: {input_path}")
        
        # Leer imágenes del directorio
        df = self.spark.read.format("binaryFile") \
            .option("recursiveFileLookup", "true") \
            .load(input_path)
        
        # Esquema para los resultados
        result_schema = StructType([
            StructField("filename", StringType(), True),
            StructField("fruit_name", StringType(), True),
            StructField("confidence", FloatType(), True),
            StructField("price_soles", FloatType(), True),
            StructField("timestamp", TimestampType(), True)
        ])
        
        # UDF para procesar imágenes
        process_image_udf = udf(self._process_single_image, result_schema)
        
        # Procesar imágenes
        results_df = df.select(
            col("path").alias("image_path"),
            col("content").alias("image_data")
        ).withColumn("result", process_image_udf(col("image_path"), col("image_data"))) \
         .select(col("result.*"))
        
        # Guardar resultados
        results_df.coalesce(1) \
            .write \
            .mode("overwrite") \
            .option("header", "true") \
            .csv(output_path)
        
        # Mostrar estadísticas
        total_images = results_df.count()
        fruit_distribution = results_df.groupBy("fruit_name").count().orderBy(desc("count"))
        avg_confidence = results_df.agg(avg("confidence")).collect()[0][0]
        
        logger.info(f"Procesadas {total_images} imágenes")
        logger.info(f"Confianza promedio: {avg_confidence:.2%}")
        
        fruit_distribution.show(15)
        
        return results_df
    
    def start_streaming_consumer(self):
        """Inicia el consumer de Kafka con Spark Streaming"""
        logger.info(f"Iniciando consumer de Kafka: {self.topic}")
        
        # Leer stream de Kafka
        kafka_stream = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_servers) \
            .option("subscribe", self.topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parsear mensajes JSON
        parsed_stream = kafka_stream.select(
            col("key").cast("string").alias("message_key"),
            from_json(col("value").cast("string"), self._get_kafka_schema()).alias("data"),
            col("timestamp").alias("kafka_timestamp")
        ).select(
            col("message_key"),
            col("data.*"),
            col("kafka_timestamp")
        )
        
        # UDF para procesar imágenes desde base64
        process_kafka_image_udf = udf(self._process_kafka_image, self._get_result_schema())
        
        # Procesar imágenes del stream
        processed_stream = parsed_stream.withColumn(
            "result", 
            process_kafka_image_udf(col("image_base64"), col("image_name"))
        ).select(
            col("message_key"),
            col("result.*"),
            col("kafka_timestamp")
        )
        
        # Escribir resultados a consola y archivo
        query = processed_stream.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .trigger(processingTime="10 seconds") \
            .start()
        
        # También escribir a archivo JSON
        file_query = processed_stream.writeStream \
            .outputMode("append") \
            .format("json") \
            .option("path", "./output/streaming_results") \
            .option("checkpointLocation", "./checkpoint/streaming") \
            .trigger(processingTime="30 seconds") \
            .start()
        
        # Esperar terminación
        try:
            query.awaitTermination()
            file_query.awaitTermination()
        except KeyboardInterrupt:
            logger.info("Deteniendo consumer...")
            query.stop()
            file_query.stop()
    
    def _process_single_image(self, image_path, image_data):
        """Procesa una imagen individual usando el modelo"""
        try:
            # Cargar modelo
            model = keras.models.load_model(self.broadcast_model_path.value)
            
            # Convertir bytes a imagen PIL
            image = Image.open(BytesIO(image_data))
            
            # Preprocesar imagen
            img_array = self._preprocess_image(image)
            
            # Predicción
            prediction = model.predict(img_array, verbose=0)
            y_class = prediction.argmax(axis=-1)[0]
            confidence = float(prediction[0][y_class])
            
            # Obtener nombre de fruta
            fruit_name = self.broadcast_labels.value[y_class].capitalize()
            
            # Obtener precio
            price = self.broadcast_precios.value.get(
                self.broadcast_labels.value[y_class], 0.0
            )
            
            return {
                'filename': os.path.basename(image_path),
                'fruit_name': fruit_name,
                'confidence': confidence,
                'price_soles': price,
                'timestamp': current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {str(e)}")
            return {
                'filename': os.path.basename(image_path) if image_path else "unknown",
                'fruit_name': "Error",
                'confidence': 0.0,
                'price_soles': 0.0,
                'timestamp': current_timestamp()
            }
    
    def _process_kafka_image(self, image_base64, image_name):
        """Procesa imagen desde Kafka (base64)"""
        try:
            # Decodificar base64
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            
            # Cargar modelo
            model = keras.models.load_model(self.broadcast_model_path.value)
            
            # Preprocesar
            img_array = self._preprocess_image(image)
            
            # Predicción
            prediction = model.predict(img_array, verbose=0)
            y_class = prediction.argmax(axis=-1)[0]
            confidence = float(prediction[0][y_class])
            
            # Obtener información
            fruit_name = self.broadcast_labels.value[y_class].capitalize()
            price = self.broadcast_precios.value.get(
                self.broadcast_labels.value[y_class], 0.0
            )
            
            return {
                'filename': image_name or "kafka_image",
                'fruit_name': fruit_name,
                'confidence': confidence,
                'price_soles': price,
                'processing_time': current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error procesando imagen Kafka: {str(e)}")
            return {
                'filename': image_name or "error",
                'fruit_name': "Error",
                'confidence': 0.0,
                'price_soles': 0.0,
                'processing_time': current_timestamp()
            }
    
    def _preprocess_image(self, image):
        """Preprocesa imagen para el modelo"""
        # Redimensionar a 224x224
        image = image.resize((224, 224))
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertir a array y normalizar
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _get_kafka_schema(self):
        """Esquema para mensajes de Kafka"""
        return StructType([
            StructField("image_name", StringType(), True),
            StructField("image_base64", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("upload_time", StringType(), True)
        ])
    
    def _get_result_schema(self):
        """Esquema para resultados de procesamiento"""
        return StructType([
            StructField("filename", StringType(), True),
            StructField("fruit_name", StringType(), True),
            StructField("confidence", FloatType(), True),
            StructField("price_soles", FloatType(), True),
            StructField("processing_time", TimestampType(), True)
        ])
    
    def stop(self):
        """Detiene la sesión de Spark"""
        if self.spark:
            self.spark.stop()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spark Consumer para Clasificación de Frutas')
    parser.add_argument('--mode', choices=['batch', 'streaming'], required=True,
                       help='Modo de procesamiento')
    parser.add_argument('--model', required=True,
                       help='Ruta al modelo entrenado (.h5)')
    parser.add_argument('--input', 
                       help='Directorio de entrada (para modo batch)')
    parser.add_argument('--output',
                       help='Directorio de salida (para modo batch)')
    parser.add_argument('--kafka-servers', default='localhost:9092',
                       help='Servidores Kafka')
    parser.add_argument('--topic', default='fruit-images',
                       help='Tópico de Kafka')
    
    args = parser.parse_args()
    
    # Crear consumer
    consumer = SparkFruitClassifier(
        model_path=args.model,
        kafka_servers=args.kafka_servers,
        topic=args.topic
    )
    
    try:
        if args.mode == 'batch':
            if not args.input or not args.output:
                print("Error: Para modo batch necesitas --input y --output")
                return
            
            consumer.process_batch_images(args.input, args.output)
            
        elif args.mode == 'streaming':
            consumer.start_streaming_consumer()
            
    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario")
    finally:
        consumer.stop()


if __name__ == "__main__":
    main()