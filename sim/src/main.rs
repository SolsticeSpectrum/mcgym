//! sim binary, boots the steel + azalea pair and serves the trainer over shm + uds

use std::path::PathBuf;
use std::process::ExitCode;

use sim::gym::Sim;
use sim::transport::Transport;

// covers the stride 4 far grid with margin
const VIEW: u8 = 3;

struct Args {
    shm:     PathBuf,
    sock:    PathBuf,
    agents:  usize,
    seed:    i64,
    spacing: i32,
    world:   Option<PathBuf>, // loaded map instead of worldgen
    spawn:   [f64; 4],        // x y z yaw for loaded maps
    mining:  bool,
}

fn parse(args: &mut dyn Iterator<Item = String>) -> Result<Args, String> {
    let mut shm     = None;
    let mut sock    = None;
    let mut agents  = None;
    let mut seed    = 0i64;
    let mut spacing = 1024i32;
    let mut world   = None;
    let mut spawn   = None;
    let mut mining  = true;

    while let Some(flag) = args.next() {
        let mut val = || args.next().ok_or_else(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--shm"     => shm     = Some(PathBuf::from(val()?)),
            "--sock"    => sock    = Some(PathBuf::from(val()?)),
            "--agents"  => agents  = Some(val()?.parse().map_err(|e| format!("--agents: {e}"))?),
            "--seed"    => seed    = val()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--spacing" => spacing = val()?.parse().map_err(|e| format!("--spacing: {e}"))?,
            "--world"   => world   = Some(PathBuf::from(val()?)),
            "--spawn"   => {
                let v: Vec<f64> = val()?
                    .split(',')
                    .map(|p| p.parse().map_err(|e| format!("--spawn: {e}")))
                    .collect::<Result<_, _>>()?;
                if v.len() != 4 {
                    return Err("--spawn wants x,y,z,yaw".into());
                }
                spawn = Some([v[0], v[1], v[2], v[3]]);
            }
            "--mining"  => mining  = val()?.parse::<u8>().map_err(|e| format!("--mining: {e}"))? != 0,
            other       => return Err(format!("unknown argument {other}")),
        }
    }

    if world.is_some() && spawn.is_none() {
        return Err("--world needs --spawn".into());
    }

    Ok(Args {
        shm:     shm.ok_or("--shm is required")?,
        sock:    sock.ok_or("--sock is required")?,
        agents:  agents.ok_or("--agents is required")?,
        seed,
        spacing,
        world,
        spawn:   spawn.unwrap_or([0.0; 4]),
        mining,
    })
}

fn run() -> Result<(), String> {
    let args = parse(&mut std::env::args().skip(1))?;
    if args.agents == 0 {
        return Err("--agents must be > 0".into());
    }

    let mut gym = if let Some(dir) = &args.world {
        eprintln!("[sim] booting {} agents on map {}", args.agents, dir.display());
        Sim::fixed(args.agents, dir,
                   [args.spawn[0], args.spawn[1], args.spawn[2]],
                   args.spawn[3] as f32, args.mining, VIEW)
    } else {
        eprintln!("[sim] booting {} agents (seed={} spacing={})", args.agents, args.seed, args.spacing);
        Sim::new(args.agents, args.seed, args.spacing, VIEW)
    };

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
            eprintln!("[sim] error: {e}");
            ExitCode::FAILURE
        }
    }
}
