import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.intent_router import IntentRouter
from core.rag_pipeline import RAGPipeline
from core.orchestrator import AgentOrchestrator

EVAL_DIR = Path(__file__).resolve().parent

class AgroNerveBenchmark:
    """Automated benchmark runner calculating routing accuracy, confusion matrices, and retrieval metrics."""

    def __init__(self):
        self.router = IntentRouter()
        self.rag = RAGPipeline()
        self.orchestrator = AgentOrchestrator()
        self.queries_file = EVAL_DIR / "test_queries.json"

    def load_queries(self) -> List[Dict[str, str]]:
        if not self.queries_file.exists():
            return []
        with open(self.queries_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(self) -> Dict[str, Any]:
        queries = self.load_queries()
        if not queries:
            return {"error": "No test queries found."}

        domains = ["disease", "pesticide", "weather", "irrigation"]
        confusion_matrix = {actual: {pred: 0 for pred in domains + ["general"]} for actual in domains}
        
        stage1_count = 0
        correct_count = 0
        total_queries = len(queries)

        domain_correct = defaultdict(int)
        domain_total = defaultdict(int)

        retrieval_hits = 0

        for q in queries:
            actual = q["domain"]
            query_text = q["query"]

            # Route
            route_res = self.router.route(query_text)
            predicted = route_res["domain"]
            
            if route_res.get("stage_used") == 1 and route_res.get("is_confident"):
                stage1_count += 1

            if actual in confusion_matrix and predicted in confusion_matrix[actual]:
                confusion_matrix[actual][predicted] += 1

            domain_total[actual] += 1
            if predicted == actual:
                correct_count += 1
                domain_correct[actual] += 1

            # Check retrieval
            chunks = self.rag.retrieve(query_text, predicted, top_k=5)
            if chunks:
                retrieval_hits += 1

        overall_accuracy = (correct_count / total_queries) * 100 if total_queries else 0.0
        stage1_rate = (stage1_count / total_queries) * 100 if total_queries else 0.0

        return {
            "total_queries": total_queries,
            "overall_accuracy_pct": round(overall_accuracy, 2),
            "keyword_stage1_rate_pct": round(stage1_rate, 2),
            "domain_accuracies": {
                d: round((domain_correct[d] / domain_total[d]) * 100, 2) if domain_total[d] else 0.0
                for d in domains
            },
            "confusion_matrix": confusion_matrix,
            "retrieval_recall_top5": round((retrieval_hits / total_queries), 2)
        }

    def print_summary(self):
        results = self.run_benchmark()
        print("=" * 60)
        print("[*] AgroNerve Benchmark Evaluation Results")
        print("=" * 60)
        print(f"Total Test Queries Evaluated: {results['total_queries']}")
        print(f"Overall Intent Routing Accuracy: {results['overall_accuracy_pct']}%")
        print(f"Stage-1 (Keyword Fast-Path) Rate: {results['keyword_stage1_rate_pct']}%")
        print(f"Retrieval Recall@5: {results['retrieval_recall_top5'] * 100}%")
        print("-" * 60)
        print("Domain-Specific Accuracies:")
        for d, acc in results["domain_accuracies"].items():
            print(f"  * {d.capitalize():12}: {acc}%")
        print("-" * 60)
        print("Confusion Matrix (Actual Rows -> Predicted Columns):")
        header = f"{'Actual':12} | " + " | ".join([f"{col[:4]:4}" for col in ["dise", "pest", "weat", "irri", "gene"]])
        print(header)
        print("-" * len(header))
        for actual, row in results["confusion_matrix"].items():
            row_str = " | ".join([f"{row.get(col, 0):4}" for col in ["disease", "pesticide", "weather", "irrigation", "general"]])
            print(f"{actual.capitalize():12} | {row_str}")
        print("=" * 60)

if __name__ == "__main__":
    runner = AgroNerveBenchmark()
    runner.print_summary()
