use std::fs;
use std::time::Duration;

use bevy::app::TaskPoolThreadAssignmentPolicy;
use bevy::core_pipeline::smaa::{Smaa, SmaaPreset};
use bevy::ecs::system::SystemId;
use bevy::platform::collections::HashSet;
use bevy::window::{PresentMode, WindowResolution, WindowTheme};
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    input::mouse::MouseMotion,
    prelude::*,
};
use bevy_framepace::{FramepacePlugin, FramepaceSettings};

use crate::chunk::CHUNK_STORE;
use crate::registry::{ChunkPos, ChunkTask, LoaderChannels, RegistriesHandle};
use crate::{chunk::Chunk, textures::DynamicTextureAtlas};

mod chunk;
mod registry;
mod textures;

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "worldgenpreview".into(),
                    window_theme: Some(WindowTheme::Dark),
                    resolution: WindowResolution::new(1500.0, 800.0),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .set(TaskPoolPlugin {
                task_pool_options: TaskPoolOptions {
                    async_compute: TaskPoolThreadAssignmentPolicy {
                        min_threads: 0,
                        max_threads: usize::MAX,
                        percent: 0.5,
                        on_thread_spawn: None,
                        on_thread_destroy: None,
                    },
                    ..Default::default()
                },
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
    .add_systems(
        Update,
        (
            ChunkTask::try_complete_sys,
            ChunkTask::load_textures_sys,
            spectator_controls,
            misc_controls,
            update_stats,
            generate_chunks_around_camera,
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
        Msaa::Off,
        Smaa {
            preset: SmaaPreset::High,
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
    mut chunk_tasks: Query<(Entity, Option<&mut ChunkTask>), With<ChunkPos>>,
) {
    commands.remove_resource::<DynamicTextureAtlas>();
    commands.remove_resource::<LoaderChannels>();
    commands.remove_resource::<RegistriesHandle>();
    commands.init_resource::<DynamicTextureAtlas>();
    state.set(AppState::Loading);

    for (entity, task) in &mut chunk_tasks {
        if let Some(mut task) = task {
            task.cancel();
        } else {
            commands.entity(entity).despawn();
        }
    }
    CHUNK_STORE.lock().unwrap().clear();
}

const RENDER_DISTANCE: i32 = 48;
const HALF_RENDER_DISTANCE: i32 = RENDER_DISTANCE / 2;

fn generate_chunks_around_camera(
    mut commands: Commands,
    cam: Query<&Transform, With<Camera3d>>,
    mut chunks: Query<(Entity, &ChunkPos, Option<&mut ChunkTask>)>,
    registries: Res<RegistriesHandle>,
) {
    let cam_pos = cam.single().unwrap().translation;
    let cam_chunk = (cam_pos / 16.0).xz().as_ivec2();
    let mut generated_chunks = HashSet::new();
    for (entity, pos, task) in &mut chunks {
        if pos.0.distance_squared(cam_chunk) > HALF_RENDER_DISTANCE.pow(2) {
            if let Some(mut task) = task {
                task.cancel();
            } else {
                commands.entity(entity).despawn();
            }
        } else {
            generated_chunks.insert(pos.0);
        }
    }
    let chunks_to_generate = (0..RENDER_DISTANCE * RENDER_DISTANCE).flat_map(|i| {
        let x = (i % RENDER_DISTANCE) - HALF_RENDER_DISTANCE + 1;
        let z = (i / RENDER_DISTANCE) - HALF_RENDER_DISTANCE + 1;
        let pos = IVec2::new(x, z) + cam_chunk;
        if pos.distance_squared(cam_chunk) <= HALF_RENDER_DISTANCE.pow(2)
            && !generated_chunks.contains(&pos)
        {
            Some(pos)
        } else {
            None
        }
    });

    for pos in chunks_to_generate {
        commands.spawn((ChunkTask::create(registries.clone(), pos), ChunkPos(pos)));
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

    let text_spans = [
        ("", 168, 255, 148, "fps"),
        ("", 150, 150, 150, "vsync"),
        ("Memory: ", 200, 200, 200, ""),
        ("", 148, 191, 255, "memory"),
        (" MB | ", 150, 150, 150, ""),
        ("Resources: ", 200, 200, 200, ""),
        ("", 168, 255, 148, "resources/ok"),
        ("", 255, 152, 148, "resources/err"),
        ("", 150, 150, 150, "resources/ldg"),
        ("| Generating: ", 200, 200, 200, ""),
        ("", 246, 255, 148, "chunks"),
        (" Chunks", 150, 150, 150, ""),
    ];
    let mut built_text_spans = Vec::new();
    for (text, r, g, b, id) in text_spans {
        let entity = commands
            .spawn((
                TextSpan::new(text),
                TextColor(Color::srgb_u8(r, g, b)),
                TextId(id),
            ))
            .id();
        built_text_spans.push(entity);
    }

    let text = commands
        .spawn((
            Text::new("FPS: "),
            stats_font(&asset_server),
            TextColor(Color::srgb_u8(200, 200, 200)),
        ))
        .add_children(&built_text_spans)
        .id();

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                padding: UiRect::axes(Val::Px(20.0), Val::Px(10.0)),
                ..Default::default()
            },
            BackgroundColor(Color::srgba_u8(0, 0, 0, 200)),
        ))
        .add_child(text);
}

#[derive(Component)]
pub struct TextId(&'static str);

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
    mut dyn_text: Query<(&mut TextSpan, &TextId)>,
    mut timer: ResMut<UpdateStatsTimer>,
    registries: Res<RegistriesHandle>,
    vsync_enabled: Res<VSyncMode>,
    time: Res<Time>,
    chunks: Query<(), With<ChunkTask>>,
) {
    timer.0.tick(time.delta());
    let span = info_span!("aquire_registries").entered();
    let status = registries.lock().as_ref().get_total_status();
    drop(span);
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
            ("chunks", _) => **text = format!("{:>3}", chunks.iter().count()),
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

type BlockMaterial = StandardMaterial;
