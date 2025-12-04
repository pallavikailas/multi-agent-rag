from deepagents import SupervisorAgent

class RAGSupervisor(SupervisorAgent):
    def route(self, state):
        if "docs" not in state:
            return "retrieve"
        if "answer" not in state:
            return "qa"
        return "summarize"
