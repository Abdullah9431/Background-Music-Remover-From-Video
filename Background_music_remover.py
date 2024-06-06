from df.enhance import enhance, init_df, load_audio, save_audio
import moviepy.editor as mpe
import os


os.chdir('/home/abdullah9431/Videos')
input_file = "video_2024-06-01_18-57-17.mp4"
output_file = "video_2024-06-01_18-48-57.mp4"

video_clip = mpe.VideoFileClip(input_file)
audio_clip = video_clip.audio # Get audio from the video
audio_clip.write_audiofile("process.wav")

model, df_state, _ = init_df(epoch='latest')
audio_clip, _ = load_audio("process.wav", sr=df_state.sr()) # Because no implementation found for audio_clip to tensor
enhanced = enhance(model, df_state, audio_clip) # 
save_audio("process.wav", enhanced, df_state.sr())

desired_audio = mpe.AudioFileClip("process.wav")
final_video = video_clip.set_audio(desired_audio) # Adding desired audio to the video
final_video.write_videofile(output_file)
