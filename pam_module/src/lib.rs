use doorman_shared::{Request, Response, AUTH_TIMEOUT_SECS, SOCKET_PATH};
use pam::{constants::*, module::PamHandle, pam_hooks};
use std::ffi::CStr;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// Connect to daemon and send authentication request
fn authenticate_user(username: &str) -> Result<bool, Box<dyn std::error::Error>> {
    // Set a connection timeout
    let socket = UnixStream::connect_timeout(
        &std::os::unix::net::SocketAddr::from_pathname(SOCKET_PATH)?,
        Duration::from_secs(AUTH_TIMEOUT_SECS),
    )?;
    
    // Set read timeout for the response
    socket.set_read_timeout(Some(Duration::from_secs(AUTH_TIMEOUT_SECS)))?;
    socket.set_write_timeout(Some(Duration::from_secs(1)))?;

    // Send authentication request
    let request = Request::Authenticate {
        username: username.to_string(),
    };
    
    let mut stream = socket;
    let request_json = serde_json::to_string(&request)?;
    writeln!(stream, "{}", request_json)?;
    stream.flush()?;

    // Read response
    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;

    let response: Response = serde_json::from_str(&response_line)?;

    match response {
        Response::Success { .. } => Ok(true),
        Response::Failure { .. } => Ok(false),
        _ => Ok(false),
    }
}

/// PAM authentication hook
pam_hooks!(PamDoorman);

pub struct PamDoorman;
impl pam::module::PamHooks for PamDoorman {
    fn sm_authenticate(pamh: &mut PamHandle, _args: Vec<&CStr>, _flags: PamFlag) -> PamResultCode {
        // Get the username from PAM
        let username = match pamh.get_user(None) {
            Ok(Some(u)) => match u.to_str() {
                Ok(s) => s,
                Err(_) => return PamResultCode::PAM_AUTH_ERR,
            },
            Ok(None) => return PamResultCode::PAM_USER_UNKNOWN,
            Err(_) => return PamResultCode::PAM_AUTH_ERR,
        };

        // Skip root authentication via doorman for safety
        if username == "root" {
            return PamResultCode::PAM_AUTH_ERR;
        }

        // Try to authenticate via the daemon
        match authenticate_user(username) {
            Ok(true) => PamResultCode::PAM_SUCCESS,
            Ok(false) => PamResultCode::PAM_AUTH_ERR,
            Err(_) => {
                // If daemon is down or timeout, fall through to next PAM module
                // Using PAM_AUTH_ERR will make it fall through in "sufficient" mode
                PamResultCode::PAM_AUTH_ERR
            }
        }
    }

    fn sm_setcred(_pamh: &mut PamHandle, _args: Vec<&CStr>, _flags: PamFlag) -> PamResultCode {
        PamResultCode::PAM_SUCCESS
    }
}

