# Keeps evolution going. SaturnCI evaluates this whenever a push lands and
# again whenever one of its jobs finishes, so starting another evolve run each
# time the last one passes is what makes a chain of them. A run that fails ends
# the chain rather than retrying in a tight loop -- a broken build would
# otherwise restart forever. Rerunning the workflow picks the chain back up,
# from the last champion in the log.
class DefaultWorkflow
  def initialize(env: ENV)
    @env = env
  end

  def perform(io: $stdout)
    return unless workflow_run.branch_name == "main"
    return unless evolve_job_runs.all?(&:passed?)
    return if SaturnCI::JobRun.list(client: saturnci_client, job_name: "evolve", status: "running").any?

    evolve_job_run = workflow_run.job_runs.create(job_name: "evolve", task_adapter_name: "shell")
    evolve_job_run.start

    io.puts "Started evolve job run: id=#{evolve_job_run.id}"
  end

  private

  def evolve_job_runs
    workflow_run.job_runs.list(job_name: "evolve")
  end

  def workflow_run
    @workflow_run ||= SaturnCI::WorkflowRun.find(
      client: saturnci_client,
      id: @env.fetch("WORKFLOW_RUN_ID")
    )
  end

  def saturnci_client
    SaturnCI::Client.new(
      SaturnCI::Credentials.new(api_token: @env.fetch("SATURNCI_ACCESS_TOKEN"))
    )
  end
end

DefaultWorkflow.new.perform if $PROGRAM_NAME == __FILE__
