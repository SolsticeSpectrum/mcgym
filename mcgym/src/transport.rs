//! Shared-memory + Unix-domain-socket bridge to the Python trainer.
//!
//! Mirrors `trainer/mcai_train/env/shm_transport.py` and the Java `McaiTransport`.
//! One mapped file, all little-endian:
//!
//! ```text
//!   [0 .. 64)                     header: magic i32, schema_version i32, n_agents i32, reserved
//!   [64 .. 64 + n*ACTION_NBYTES)  ACTION region (Python writes, gym reads)
//!   [.. + n*OBS_NBYTES)           OBS region    (gym writes, Python reads)
//! ```
//!
//! Stepping is paced by a one-byte command / one-byte reply over the socket: the
//! client sends RESET/STEP/CLOSE, the gym does the work and writes the OBS region,
//! then replies OK. The blocking read paces the whole gym to the driver.

use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use memmap2::MmapMut;

use crate::schema::{ACTION_NBYTES, OBS_NBYTES, SCHEMA_VERSION};

pub const HEADER_NBYTES: usize = 64;
pub const MAGIC: i32 = 0x4D43_4149; // 'MCAI'

pub const CMD_RESET: u8 = 1;
pub const CMD_STEP: u8 = 2;
pub const CMD_CLOSE: u8 = 3;
pub const REPLY_OK: u8 = 1;

pub const READY_LINE: &str = "MCAI_TRANSPORT_READY";

/// The gym behaviour the transport drives. The transport hands disjoint slices:
/// `actions` is the whole ACTION region (`n_agents * ACTION_NBYTES`, read-only),
/// `obs` is the whole OBS region (`n_agents * OBS_NBYTES`, write it fully).
pub trait Gym {
    fn reset(&mut self, obs: &mut [u8]);
    fn step(&mut self, actions: &[u8], obs: &mut [u8]);
}

pub struct Transport {
    mmap: MmapMut,
    listener: UnixListener,
    sock_path: PathBuf,
    client: Option<UnixStream>,
    n_agents: usize,
    obs_off: usize,
}

impl Transport {
    /// Create + size + map the shm file, write the header, and bind the UDS listener.
    /// No silent fallbacks: fails fast if paths are unusable.
    pub fn create(shm_path: &Path, sock_path: &Path, n_agents: usize) -> io::Result<Self> {
        let obs_off = HEADER_NBYTES + n_agents * ACTION_NBYTES;
        let total = obs_off + n_agents * OBS_NBYTES;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(shm_path)?;
        file.set_len(total as u64)?;
        // SAFETY: we own the file and keep it mapped for the Transport's lifetime.
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Header (little-endian): magic, schema_version, n_agents, rest zeroed by set_len.
        mmap[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        mmap[4..8].copy_from_slice(&SCHEMA_VERSION.to_le_bytes());
        mmap[8..12].copy_from_slice(&(n_agents as i32).to_le_bytes());

        if let Some(parent) = sock_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _ = std::fs::remove_file(sock_path);
        let listener = UnixListener::bind(sock_path)?;

        Ok(Self {
            mmap,
            listener,
            sock_path: sock_path.to_path_buf(),
            client: None,
            n_agents,
            obs_off,
        })
    }

    /// Announce readiness on stdout so the launcher (`gym_process.py`) can detect it.
    pub fn signal_ready(&self) {
        println!("{READY_LINE}");
        let _ = io::stdout().flush();
    }

    /// Block until the driver connects.
    pub fn accept(&mut self) -> io::Result<()> {
        let (stream, _addr) = self.listener.accept()?;
        self.client = Some(stream);
        Ok(())
    }

    /// Disjoint slices into the mapped file: (whole ACTION region, whole OBS region).
    fn regions(&mut self) -> (&[u8], &mut [u8]) {
        let (head_and_actions, obs) = self.mmap.split_at_mut(self.obs_off);
        (&head_and_actions[HEADER_NBYTES..], obs)
    }

    /// Run the command loop until CLOSE or the socket drops. `accept` must precede this.
    pub fn serve<G: Gym>(&mut self, gym: &mut G) -> io::Result<()> {
        let mut stream = self
            .client
            .take()
            .ok_or_else(|| io::Error::other("serve() called before accept()"))?;
        let mut cmd = [0u8; 1];
        loop {
            if let Err(e) = stream.read_exact(&mut cmd) {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    return Ok(()); // driver hung up
                }
                return Err(e);
            }
            match cmd[0] {
                CMD_RESET => {
                    let (_actions, obs) = self.regions();
                    gym.reset(obs);
                    stream.write_all(&[REPLY_OK])?;
                }
                CMD_STEP => {
                    let (actions, obs) = self.regions();
                    gym.step(actions, obs);
                    stream.write_all(&[REPLY_OK])?;
                }
                CMD_CLOSE => {
                    let _ = stream.write_all(&[REPLY_OK]);
                    return Ok(());
                }
                other => {
                    return Err(io::Error::other(format!("unknown command byte {other}")));
                }
            }
        }
    }

    pub fn n_agents(&self) -> usize {
        self.n_agents
    }
}

impl Drop for Transport {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.sock_path);
    }
}
