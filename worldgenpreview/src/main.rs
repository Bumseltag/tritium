use std::fs;
use std::time::Duration;

use bevy::ecs::system::SystemId;
use bevy::window::{PresentMode, WindowResolution, WindowTheme};
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::MouseMotion,
    prelude::*,
};
use bevy_framepace::{FramepacePlugin, FramepaceSettings};

use crate::registry::{ChunkTask, LoaderChannels, RegistriesHandle};
use crate::{chunk::Chunk, textures::DynamicTextureAtlas};

mod chunk;
mod registry;
mod textures;

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "worldgenpreview".into(),
                window_theme: Some(WindowTheme::Dark),
                resolution: WindowResolution::new(1500.0, 800.0),
                ..Default::default()
            }),
            ..Default::default()
        }),
        FramepacePlugin,
        FrameTimeDiagnosticsPlugin::default(),
    ))
    .init_resource::<DynamicTextureAtlas>()
    .init_resource::<UpdateStatsTimer>()
    .init_resource::<VSyncMode>()
    .init_state::<AppState>()
    .add_systems(Startup, (setup_scene, setup_stats))
    .add_systems(OnEnter(AppState::Loading), ChunkTask::init_sys)
    .add_systems(OnEnter(AppState::Main), spawn_example_chunks)
    .add_systems(
        Update,
        (
            ChunkTask::try_complete_sys,
            ChunkTask::load_textures_sys,
            spectator_controls,
            misc_controls,
            update_stats,
        )
            .run_if(in_state(AppState::Main)),
    );
    app.run();

    fs::remove_dir_all("worldgenpreview_cache").unwrap();
}

#[derive(States, Default, Debug, Hash, PartialEq, Eq, Clone)]
pub enum AppState {
    #[default]
    Loading,
    Main,
}

fn setup_scene(mut commands: Commands) {
    info!("Chunk size: {}", size_of::<Chunk>());

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

    let reload_system = commands.register_system(reload);
    commands.insert_resource(ReloadSystem(reload_system));
}

#[derive(Resource)]
pub struct ReloadSystem(SystemId);

fn reload(
    mut commands: Commands,
    mut state: ResMut<NextState<AppState>>,
    mut chunk_tasks: Query<&mut ChunkTask>,
) {
    commands.remove_resource::<DynamicTextureAtlas>();
    commands.remove_resource::<LoaderChannels>();
    commands.remove_resource::<RegistriesHandle>();
    commands.init_resource::<DynamicTextureAtlas>();
    state.set(AppState::Loading);

    for mut task in &mut chunk_tasks {
        task.cancel();
    }
}

fn spawn_example_chunks(mut commands: Commands, registries: Res<RegistriesHandle>) {
    let registries = registries.clone();
    commands.spawn_batch((0..24 * 24).map(move |i| {
        let x = i % 24;
        let z = i / 24;
        ChunkTask::create(IVec2::new(x, z), registries.clone())
    }));
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

    let mut move_speed = 20.0;
    if keys.pressed(KeyCode::ControlLeft) {
        move_speed *= 5.0;
    }
    if keys.pressed(KeyCode::AltLeft) {
        move_speed *= 5.0;
    }

    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
        transform.translation += direction * move_speed * time.delta_secs();
    }
}

fn misc_controls(
    mut commands: Commands,
    mut windows: Query<&mut Window>,
    mut vsync_enabled: ResMut<VSyncMode>,
    mut framepace: ResMut<FramepaceSettings>,
    keys: Res<ButtonInput<KeyCode>>,
    reload_system: Res<ReloadSystem>,
) {
    let mut window = windows.single_mut().unwrap();
    if keys.just_pressed(KeyCode::KeyV) {
        if vsync_enabled.0 {
            window.present_mode = PresentMode::AutoNoVsync;
            framepace.limiter = bevy_framepace::Limiter::Off;
        } else {
            window.present_mode = PresentMode::AutoVsync;
            framepace.limiter = bevy_framepace::Limiter::Auto;
        }
        vsync_enabled.0 = !vsync_enabled.0;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        commands.run_system(reload_system.0);
    }
}

fn setup_stats(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            display: Display::Block,
            bottom: Val::Px(0.0),
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            padding: UiRect::axes(Val::Px(20.0), Val::Px(10.0)),
            ..Default::default()
        },
        BackgroundColor(Color::srgba_u8(0, 0, 0, 200)),
        children![(
            Text::new("Uses spectator controls | [RMB] Pan Camera | [Ctrl] move fast | [Ctrl+Alt] move super fast\n[V] toggle VSync | [R] reload resources"),
            stats_font(&asset_server),
            TextColor(Color::srgb_u8(200, 200, 200)),
        )],
    ));
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            padding: UiRect::axes(Val::Px(20.0), Val::Px(10.0)),
            ..Default::default()
        },
        BackgroundColor(Color::srgba_u8(0, 0, 0, 200)),
        children![(
            Text::new("FPS: "),
            stats_font(&asset_server),
            TextColor(Color::srgb_u8(200, 200, 200)),
            children![
                (
                    TextSpan::new("---.-"),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(168, 255, 148)),
                    DynText("fps"),
                ),
                (
                    TextSpan::new("             | "),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(150, 150, 150)),
                    DynText("vsync"),
                ),
                (
                    TextSpan::new("Memory: "),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(200, 200, 200)),
                ),
                (
                    TextSpan::new("----.-"),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(148, 191, 255)),
                    DynText("memory")
                ),
                (
                    TextSpan::new(" MB | "),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(150, 150, 150))
                ),
                (
                    TextSpan::new("Resources: "),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(200, 200, 200)),
                ),
                (
                    TextSpan::new("--- Ok "),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(168, 255, 148)),
                    DynText("resources/ok")
                ),
                (
                    TextSpan::new("--- Err "),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(255, 152, 148)),
                    DynText("resources/err")
                ),
                (
                    TextSpan::new("--- Ldg "),
                    stats_font(&asset_server),
                    TextColor(Color::srgb_u8(150, 150, 150)),
                    DynText("resources/ldg")
                ),
            ]
        )],
    ));
}

#[derive(Component)]
pub struct DynText(&'static str);

#[derive(Resource)]
pub struct UpdateStatsTimer(Timer);

impl Default for UpdateStatsTimer {
    fn default() -> Self {
        Self(Timer::new(Duration::from_secs(1), TimerMode::Repeating))
    }
}

#[derive(Resource)]
pub struct VSyncMode(bool);

impl Default for VSyncMode {
    fn default() -> Self {
        Self(true)
    }
}

fn update_stats(
    diagnostics: Res<DiagnosticsStore>,
    mut dyn_text: Query<(&mut TextSpan, &DynText)>,
    mut timer: ResMut<UpdateStatsTimer>,
    registries: Res<RegistriesHandle>,
    vsync_enabled: Res<VSyncMode>,
    time: Res<Time>,
) {
    timer.0.tick(time.delta());
    let status = registries.lock_blocking().get_total_status();
    for (mut text, id) in &mut dyn_text {
        match (id.0, timer.0.just_finished()) {
            ("fps", true) => {
                if let Some(fps) = diagnostics
                    .get(&FrameTimeDiagnosticsPlugin::FPS)
                    .and_then(|v| v.smoothed())
                {
                    **text = format!("{fps:>5.1}");
                } else {
                    **text = "---.-".into();
                }
            }
            ("vsync", _) => {
                **text = if vsync_enabled.0 {
                    " (VSync on)  | ".into()
                } else {
                    " (VSync off) | ".into()
                };
            }
            ("memory", true) => {
                if let Some(memory_stats) = memory_stats::memory_stats() {
                    **text = format!("{:>7.1}", memory_stats.virtual_mem as f32 / 1024.0 / 1024.0);
                } else {
                    **text = "----.-".into();
                }
            }
            ("resources/ok", _) => **text = format!("{:>3} Ok ", status.ok),
            ("resources/err", _) => **text = format!("{:>3} Err ", status.errs),
            ("resources/ldg", _) => **text = format!("{:>3} Ldg ", status.loading),
            _ => {}
        }
    }
}

fn stats_font(asset_server: &AssetServer) -> TextFont {
    TextFont {
        font: asset_server.load("JetBrainsMono-Regular.ttf"),
        font_size: 20.0,
        ..Default::default()
    }
}
