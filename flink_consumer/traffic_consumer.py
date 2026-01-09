import json
import joblib
import hashlib
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.common.time import Time
from pyflink.datastream.functions import ProcessWindowFunction
from pyflink.common.watermark_strategy import WatermarkStrategy

# ================= CONFIG =================
KAFKA_TOPIC = "traffic-data"
KAFKA_BROKERS = "kafka:29092"
MODEL_PATH = "/opt/flink/jobs/rerouting_kmeans.pkl"
WINDOW_SECONDS = 30
TOP_K_ALTERNATIVES = 2

# 🔥 MUST MATCH KAFKA PARTITIONS
NUM_BUCKETS = 3

# ================= STABLE HASH =================
def stable_bucket(street: str) -> int:
    return int(hashlib.md5(street.encode()).hexdigest(), 16) % NUM_BUCKETS

# ================= WINDOW FUNCTION =================
class MLBasedRerouting(ProcessWindowFunction):

    def open(self, runtime_context):
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.cluster_labels = bundle.get("cluster_labels", {})
        self.subtask = runtime_context.get_index_of_this_subtask()
        print(f"✅ ML model loaded in subtask {self.subtask}")

    def process(self, key, context, elements):
        street_stats = {}

        for street, severity in elements:
            street_stats.setdefault(street, {"sum": 0.0, "count": 0})
            street_stats[street]["sum"] += severity
            street_stats[street]["count"] += 1

        streets, loads = [], []

        for street, stats in street_stats.items():
            avg = stats["sum"] / stats["count"]
            load = avg * stats["count"]
            streets.append(street)
            loads.append([load])

        if not streets:
            return

        if self.scaler:
            loads = self.scaler.transform(loads)

        clusters = self.model.predict(loads)

        enriched = list(zip(streets, loads, clusters))
        sorted_by_load = sorted(enriched, key=lambda x: x[1][0])

        for street, load, cluster in enriched:
            label = self.cluster_labels.get(cluster, "UNKNOWN")

            if label == "HIGH":
                alternatives = [
                    s for s, l, c in sorted_by_load if s != street
                ][:TOP_K_ALTERNATIVES]

                yield (
                    f"[Subtask-{self.subtask}] 🚨 HIGH | {street} | "
                    f"Load={load[0]:.2f} | Reroute → {', '.join(alternatives)}"
                )

            elif label == "MEDIUM":
                yield (
                    f"[Subtask-{self.subtask}] ⚠️ MEDIUM | {street} | "
                    f"Load={load[0]:.2f}"
                )

            else:
                yield (
                    f"[Subtask-{self.subtask}] 🟢 LOW | {street} | "
                    f"Load={load[0]:.2f}"
                )


# ================= MAIN =================
def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(NUM_BUCKETS)

    source = KafkaSource.builder() \
        .set_bootstrap_servers(KAFKA_BROKERS) \
        .set_topics(KAFKA_TOPIC) \
        .set_group_id("flink-ml-group") \
        .set_starting_offsets(KafkaOffsetsInitializer.latest()) \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

    stream = env.from_source(
        source,
        WatermarkStrategy.no_watermarks(),
        "Kafka Source"
    )

    def normalize_event(x):
        try:
            d = json.loads(x)
            street = d.get("Street") or d.get("street")
            severity = d.get("Severity") or d.get("severity")
            if street is None or severity is None:
                return None
            return (street, float(severity))
        except:
            return None

    traffic = (
        stream
        .map(normalize_event, Types.TUPLE([Types.STRING(), Types.FLOAT()]))
        .filter(lambda x: x is not None)
    )

    traffic \
        .key_by(lambda x: stable_bucket(x[0]), Types.INT()) \
        .window(TumblingProcessingTimeWindows.of(Time.seconds(WINDOW_SECONDS))) \
        .process(MLBasedRerouting(), Types.STRING()) \
        .print() \
        .set_parallelism(NUM_BUCKETS)

    env.execute("Parallel ML-Based Traffic Rerouting")

if __name__ == "__main__":
    main()
