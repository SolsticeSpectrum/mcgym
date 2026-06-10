//! mcgym binary, boots the Pumpkin backed world and serves the trainer over shm + uds

use std::path::PathBuf;
use std::process::ExitCode;

use mcgym::gym::GymState;
use mcgym::transport::Transport;

struct Args {
    shm:     PathBuf,
    sock:    PathBuf,
    agents:  usize,
    seed:    i64,
    spacing: i32,
}

fn parse(args: &mut dyn Iterator<Item = String>) -> Result<Args, String> {
    let mut shm     = None;
    let mut sock    = None;
    let mut agents  = None;
    let mut seed    = 0i64;
    let mut spacing = 1024i32;

    while let Some(flag) = args.next() {
        let mut val = || args.next().ok_or_else(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--shm"     => shm     = Some(PathBuf::from(val()?)),
            "--sock"    => sock    = Some(PathBuf::from(val()?)),
            "--agents"  => agents  = Some(val()?.parse().map_err(|e| format!("--agents: {e}"))?),
            "--seed"    => seed    = val()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--spacing" => spacing = val()?.parse().map_err(|e| format!("--spacing: {e}"))?,
            other       => return Err(format!("unknown argument {other}")),
        }
    }

    Ok(Args {
        shm:     shm.ok_or("--shm is required")?,
        sock:    sock.ok_or("--sock is required")?,
        agents:  agents.ok_or("--agents is required")?,
        seed,
        spacing,
    })
}

fn run() -> Result<(), String> {
    let args = parse(&mut std::env::args().skip(1))?;
    if args.agents == 0 {
        return Err("--agents must be > 0".into());
    }

    eprintln!("[mcgym] booting {} agents (seed={} spacing={})", args.agents, args.seed, args.spacing);
    let mut gym = GymState::new(args.agents, args.seed, args.spacing);

    let mut transport = Transport::create(&args.shm, &args.sock, args.agents)
        .map_err(|e| format!("transport setup failed: {e}"))?;
    transport.signal_ready();
    transport.accept().map_err(|e| format!("accept failed: {e}"))?;
    transport.serve(&mut gym).map_err(|e| format!("serve failed: {e}"))
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
