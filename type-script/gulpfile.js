var gulp = require('gulp');
var typescript = require('gulp-typescript');
var tsproject = typescript.createProject('tsconfig.json', function() {
    typescript: require('typescript')
});

// Compile .ts to .js.
gulp.task('compile:ts', function() {
    return gulp.src(['src/ts/*.ts'])
        .pipe(typescript(tsproject))
        .js
        .pipe(gulp.dest('dst/js/'));
});

// Clean.
var del = require('del');
gulp.task('clean:dst', function() {
    return del.sync(['dst/*']);
});

// Copy .html from src to dst.
gulp.task('copy:html', function() {
    return gulp.src(['src/*.html'])
        .pipe(gulp.dest('dst/'));
});

// Detect src dir modification and compile, copy.
gulp.task('watch', function() {
    gulp.watch(
        [
            'src/ts/*.ts',
            'src/*.html'
        ],
        [
            'compile:ts',
            'copy:html'
        ]);
});

// Reload if dst is updated.
var browserSync = require('browser-sync').create();
gulp.task('server', function() {
    browserSync.init({
        server: {
            baseDir: 'dst'
        },
        files: ['dst/*']
    });
});

// Copy bower components.
gulp.task('copy:bower', function() {
    return gulp.src(
        ['src/js/bower_components/**'], // Copy files.
        { base: 'src/js' } // Base of directory reconstruction.
    )
    .pipe(gulp.dest('dst/js/'));
});

// Default task.
gulp.task('default', [
    'clean:dst',
    'compile:ts',
    'copy:html',
    'copy:bower',
    'server',
    'watch'
]);

