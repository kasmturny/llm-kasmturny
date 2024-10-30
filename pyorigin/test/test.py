from pyorigin.core.base_agent import Kafka

kafka = Kafka()
kafka.produce("test", "hello world")