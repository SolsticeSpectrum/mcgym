//! mcgym binary: parse the run parameters, boot the Pumpkin-backed world + agents, map the
//! transport, and serve the RESET/STEP/CLOSE loop. Prints `MCAI_TRANSPORT_READY` once live.
//!
//! Usage: mcgym --shm PATH --sock PATH --agents N [--seed S] [--arena wild] [--curriculum ""]

use std::path::PathBuf;
use std::process::ExitCode;

use mcgym::gym::GymState;
use mcgym::transport::Transport;

struct Args {
    shm: PathBuf,
    sock: PathBuf,
    agents: usize,
    seed: i64,
    spacing: i32,
}

fn parse_args() -> Result<Args, String> {
    let mut shm = None;
    let mut sock = None;
    let mut agents = None;
    let mut seed = 0i64;
    let mut spacing = 1024i32;

    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--shm" => shm = Some(PathBuf::from(val()?)),
            "--sock" => sock = Some(PathBuf::from(val()?)),
            "--agents" => agents = Some(val()?.parse().map_err(|e| format!("--agents: {e}"))?),
            "--seed" => seed = val()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--spacing" => spacing = val()?.parse().map_err(|e| format!("--spacing: {e}"))?,
            // Accepted for launcher compatibility; not yet differentiated.
            "--arena" | "--curriculum" => {
                let _ = val()?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    // Fail fast — no silent fallbacks for required wiring.
    Ok(Args {
        shm: shm.ok_or("--shm is required")?,
        sock: sock.ok_or("--sock is required")?,
        agents: agents.ok_or("--agents is required")?,
        seed,
        spacing,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.agents == 0 {
        return Err("--agents must be > 0".into());
    }

    eprintln!(
        "[mcgym] booting {} agents (seed={}, spacing={})…",
        args.agents, args.seed, args.spacing
    );
    let mut gym = GymState::new(args.agents, args.seed, args.spacing);

    let mut transport = Transport::create(&args.shm, &args.sock, args.agents)
        .map_err(|e| format!("transport setup failed: {e}"))?;
    transport.signal_ready();
    transport.accept().map_err(|e| format!("accept failed: {e}"))?;
    transport.serve(&mut gym).map_err(|e| format!("serve failed: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[mcgym] error: {e}");
            ExitCode::FAILURE
        }
    }
}
