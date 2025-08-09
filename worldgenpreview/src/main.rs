use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, input::mouse::MouseMotion, prelude::*};
#[cfg(feature = "log_frametime")]
use bevy::{diagnostic::LogDiagnosticsPlugin, window::PresentMode};
use bevy_framepace::FramepacePlugin;

use crate::{
    chunk::Chunk,
    registry::{LoaderChannels, Registries, ToLoader},
    textures::DynamicTextureAtlas,
};

mod chunk;
mod registry;
mod textures;

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
        FramepacePlugin,
        FrameTimeDiagnosticsPlugin::default(),
        #[cfg(feature = "log_frametime")]
        LogDiagnosticsPlugin::default(),
    ))
    .init_resource::<DynamicTextureAtlas>()
    .add_systems(Startup, (Registries::init_sys, setup).chain())
    .add_systems(Update, (Registries::update_sys, spectator_controls));
    app.run();
}
fn setup(
    mut commands: Commands,
    channels: Res<LoaderChannels>,
    #[cfg(feature = "log_frametime")] mut window: Query<&mut Window>,
) {
    #[cfg(feature = "log_frametime")]
    {
        let mut window = window.single_mut().unwrap();
        window.present_mode = PresentMode::AutoNoVsync;
    }

    println!("Chunk size: {}", size_of::<Chunk>());

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(16.0, 64.0, 16.0).looking_at(Vec3::new(0.0, 64.0, 0.0), Dir3::Y),
        AmbientLight {
            color: Color::srgb_u8(255, 255, 255),
            brightness: 1000.0,
            ..Default::default()
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -std::f32::consts::FRAC_PI_4,
            -std::f32::consts::FRAC_PI_4,
            0.0,
        )),
    ));

    for x in 0..16 {
        for z in 0..16 {
            channels
                .to_loader
                .send_blocking(ToLoader::GenChunk(IVec2::new(x, z)))
                .unwrap();
        }
    }
}

fn spectator_controls(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut query: Query<&mut Transform, With<Camera>>,
    mut windows: Query<&mut Window>,
) {
    let mut transform = query.single_mut().unwrap();

    let sensitivity = 0.3;
    let mut window = windows.single_mut().unwrap();

    if mouse_buttons.pressed(MouseButton::Right) {
        window.cursor_options.visible = false;

        let (mut yaw, mut pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);

        for event in mouse_motion_events.read() {
            yaw -= event.delta.x * sensitivity * time.delta_secs();
            pitch -= event.delta.y * sensitivity * time.delta_secs();
        }

        pitch = pitch.clamp(-1.54, 1.54);

        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    } else {
        window.cursor_options.visible = true;
    }

    let mut direction = Vec3::ZERO;
    let forward = transform.forward().normalize();
    let right = transform.right().normalize();
    let up = Vec3::Y;

    if keys.pressed(KeyCode::KeyW) {
        direction += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        direction -= forward;
    }
    if keys.pressed(KeyCode::KeyA) {
        direction -= right;
    }
    if keys.pressed(KeyCode::KeyD) {
        direction += right;
    }
    if keys.pressed(KeyCode::Space) {
        direction += up;
    }
    if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        direction -= up;
    }

    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
        transform.translation += direction * 10.0 * time.delta_secs();
    }
}
