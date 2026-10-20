# Runs worlds for a while, starting from the champion the last evolve run in
# this chain found. A chain is one workflow run's worth of evolve runs: a new
# push starts fresh rather than inheriting what earlier pushes bred. There is
# no snapshot file: the headless binary prints a `champion:` line every time
# it finds a better genome, and the next run reads the last one out of the
# previous run's log. The champion is reported once a minute as
# well as when a world ends, so a run that crashes or is cut off has still
# said what it found, and the next one carries on regardless of how this one
# ended.

require "json"

class EvolveJob
  # Simulated time, at roughly fifteen times real time on SaturnCI: a quarter
  # of an hour or so. Running out of ticks is what makes the last world -- which
  # may be the only world, if it never dies -- report its genome; a run killed
  # from outside says nothing about the world it was in the middle of.
  TICKS = "1000000"
  # Wall-clock time. Big worlds run far slower than the ticks assume, so this
  # is what usually ends a run; the next one picks up from the last champion
  # reported, losing at most a minute.
  DURATION = "1h"
  # Much bigger than a screen. More room means more critters and more pellets
  # per world, so there is more for selection to work on.
  SIZE = "6000x4000"
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

    command = ["timeout", DURATION, "./target/release/headless", "--ticks", TICKS, "--size", SIZE]
    command += ["--genome", genome] if genome
    system(*command)

    [0, TIMED_OUT].include?($?.exitstatus) ? 0 : 1
  end

  # The last champion a log reported: a genome, or "none" where the run
  # flushed a champion that failed and had found nothing better by the end.
  def self.last_champion(log)
    log.to_s.scan(/champion: ([01]+|none)/).flatten.last
  end

  private

  # The most recent earlier evolve run in this chain that left a champion
  # behind. One that died before finding anything is passed over for the one
  # before it; one that flushed a failed champion and found nothing better
  # ends the search, so the chain starts fresh rather than going back to the
  # champion that failed.
  def previous_champion
    previous_job_runs.each do |job_run|
      get("/api/v1/job_runs/#{job_run.fetch("id")}").dig("_links", "tasks").each do |task_link|
        genome = self.class.last_champion(get(task_link.fetch("href")).fetch("system_logs"))
        return nil if genome == "none"
        return genome if genome
      end
    end
    nil
  end

  # The workflow passes its id as the chain id, which arrives as CHAIN_ID.
  def previous_job_runs
    get("/api/v1/job_runs?job_name=evolve&workflow_run_id=#{@env.fetch("CHAIN_ID")}")
      .reject { |job_run| job_run.fetch("id").to_s == @env.fetch("JOB_RUN_ID") }
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
