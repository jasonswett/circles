# Runs worlds for a few hours, starting from the champion the last evolve run
# found. There is no snapshot file: the headless binary prints a `champion:`
# line every time it finds a better genome, and the next run reads the last
# one out of the previous run's log. A run that crashes has still said what it
# found, so the next one carries on regardless of how this one ended.

require "json"

class EvolveJob
  REPOSITORY = "jasonswett/circles"
  # Wall-clock time, not simulated time. `timeout` ends the run from outside,
  # and every champion it found is already in the log by then.
  DURATION = "4h"
  TIMED_OUT = 124

  def initialize(io: $stdout, env: ENV)
    @io = io
    @env = env
  end

  def perform
    system("cargo build --release --no-default-features --bin headless") or return 1

    genome = previous_champion
    if genome
      @io.puts "Resuming from the previous run's champion"
    else
      @io.puts "No previous champion; starting fresh"
    end

    command = ["timeout", DURATION, "./target/release/headless"]
    command += ["--genome", genome] if genome
    system(*command)

    [0, TIMED_OUT].include?($?.exitstatus) ? 0 : 1
  end

  def self.last_champion(log)
    log.to_s.scan(/^champion: ([01]+)$/).flatten.last
  end

  private

  # The most recent earlier evolve run that left a champion behind. One that
  # died before finding anything is passed over for the one before it.
  def previous_champion
    previous_job_runs.each do |job_run|
      get("/api/v1/job_runs/#{job_run.fetch("id")}").dig("_links", "tasks").each do |task_link|
        genome = self.class.last_champion(get(task_link.fetch("href")).fetch("system_logs"))
        return genome if genome
      end
    end
    nil
  end

  def previous_job_runs
    get("/api/v1/job_runs?job_name=evolve")
      .reject { |job_run| job_run.fetch("id").to_s == @env.fetch("JOB_RUN_ID") }
      .select { |job_run| job_run.dig("repository", "github_repo_full_name") == REPOSITORY }
  end

  def get(path)
    JSON.parse(client.get(path.sub(%r{\Ahttps?://[^/]+}, "")).body)
  end

  def client
    @client ||= SaturnCI::Client.new(
      SaturnCI::Credentials.new(api_token: @env.fetch("SATURNCI_ACCESS_TOKEN"))
    )
  end
end

if $PROGRAM_NAME == __FILE__
  $stdout.sync = true
  exit EvolveJob.new.perform
end
