"""WE7 agentic layer — the decision & action layer built on top of the RAG tool.

The flow, one module per step:

    Action
      -> action_validation.validate_action      (Part 1)
      -> context_builder.build_context          (Part 2)  -> VerificationContext
      -> policy_tool.PolicyRetrievalTool        (Part 3)  -> [RetrievedPolicy]   (RAG as a tool)
      -> verifier_agent.VerifierAgent           (Part 4)  -> Verdict
      -> solution_agent.SolutionAgent           (Part 5)  -> [Solution]          (one subagent / problem)
      -> display_agent.run_display_agent        (Part 6)  -> [UIAction]          (acts on the UI via tools)

``pipeline.VerifierPipeline`` wires these together (provided — you fill the steps).
"""
